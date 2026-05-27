import argparse
import json
from pathlib import Path

import numpy as np
import torch
import ot

from data_utils import OpenSiteRec, split
from eval_utils import PrecisionRecall_atK, NDCG_atK, get_label


BASELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASELINE_DIR.parent
ALL_CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_city", default="Singapore")
    parser.add_argument("--all_targets", type=int, default=0)
    parser.add_argument("--source_cities", nargs="*", default=None)
    parser.add_argument("--model", default="LightGCN")
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gw_eps", type=float, default=1e-9)
    parser.add_argument("--gw_max_iter", type=int, default=300)
    parser.add_argument("--gw_tol", type=float, default=1e-9)
    parser.add_argument("--gw_solver", default="classic", choices=["classic", "entropic"])
    parser.add_argument("--gammas", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metrics_out", default="")
    parser.add_argument("--per_source_gamma", type=int, default=1)
    parser.add_argument("--max_coord_iter", type=int, default=5)
    parser.add_argument("--allow_zero_gamma", type=int, default=1)
    parser.add_argument("--prescreen_sources", type=int, default=1)
    parser.add_argument("--min_improv_ndcg", type=float, default=0.0)
    parser.add_argument("--score_norm", default="none", choices=["none", "zscore"])
    parser.add_argument("--score_space", default="raw", choices=["raw", "sigmoid"])
    return parser.parse_args()


def city_split_dir(city):
    return DATA_DIR / city / "split"


def load_dataset(city, threshold, device):
    class Args:
        pass

    args = Args()
    args.city = city
    args.device = device
    split(city, threshold)
    return OpenSiteRec(args)


def load_embeddings(city, model):
    split_dir = city_split_dir(city)
    u_file = split_dir / f"{model}_U_emb.pt"
    v_file = split_dir / f"{model}_V_emb.pt"
    if not u_file.exists() or not v_file.exists():
        raise FileNotFoundError(
            f"Missing embeddings for {city}: {u_file.name} / {v_file.name}. "
            f"Run main.py with --city {city} --model {model} --save 1 first."
        )
    u = torch.load(u_file, map_location="cpu").detach().float().numpy()
    v = torch.load(v_file, map_location="cpu").detach().float().numpy()
    return u, v


def get_topk_and_metrics(ratings, dataset, topk):
    test_dict = {u: sorted(set(items)) for u, items in dataset.testDict.items()}
    all_pos = dataset.allPos
    users = list(test_dict.keys())
    items = [test_dict[u] for u in users]

    rec_sum = 0.0
    ndcg_sum = 0.0
    for start in range(0, len(users), 512):
        batch_users = users[start:start + 512]
        batch_pos = [all_pos[u] for u in batch_users]
        batch_items = [items[u] for u in batch_users]

        batch_scores = ratings[batch_users].copy()
        for i, pos_items in enumerate(batch_pos):
            batch_scores[i, pos_items] = -1e10

        idx = np.argpartition(-batch_scores, kth=topk - 1, axis=1)[:, :topk]
        topk_scores = np.take_along_axis(batch_scores, idx, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        topk_idx = np.take_along_axis(idx, order, axis=1)

        r = get_label(batch_items, topk_idx)
        _, batch_rec = PrecisionRecall_atK(batch_items, r, topk)
        batch_ndcg = NDCG_atK(batch_items, r, topk)
        rec_sum += batch_rec * len(batch_users)
        ndcg_sum += batch_ndcg * len(batch_users)

    rec = rec_sum / len(users)
    ndcg = ndcg_sum / len(users)
    return rec, ndcg


def pairwise_dist(x):
    return ot.dist(x, x, metric="euclidean") ** 2


def transport_and_project(src_u, src_v, tgt_u, tgt_v, eps):
    c_src_u = pairwise_dist(src_u)
    c_tgt_u = pairwise_dist(tgt_u)
    c_src_v = pairwise_dist(src_v)
    c_tgt_v = pairwise_dist(tgt_v)

    p_u = np.full(src_u.shape[0], 1.0 / src_u.shape[0])
    q_u = np.full(tgt_u.shape[0], 1.0 / tgt_u.shape[0])
    p_v = np.full(src_v.shape[0], 1.0 / src_v.shape[0])
    q_v = np.full(tgt_v.shape[0], 1.0 / tgt_v.shape[0])

    if args_global.gw_solver == "entropic":
        t_u = ot.gromov.entropic_gromov_wasserstein(
            c_src_u, c_tgt_u, p_u, q_u, "square_loss", epsilon=eps,
            max_iter=args_global.gw_max_iter, tol=args_global.gw_tol, verbose=False
        )
        t_v = ot.gromov.entropic_gromov_wasserstein(
            c_src_v, c_tgt_v, p_v, q_v, "square_loss", epsilon=eps,
            max_iter=args_global.gw_max_iter, tol=args_global.gw_tol, verbose=False
        )
    else:
        t_u = ot.gromov.gromov_wasserstein(
            c_src_u, c_tgt_u, p_u, q_u, "square_loss", verbose=False
        )
        t_v = ot.gromov.gromov_wasserstein(
            c_src_v, c_tgt_v, p_v, q_v, "square_loss", verbose=False
        )

    # Barycentric projection: normalize by target marginals.
    proj_u = (t_u.T @ src_u) / q_u[:, None]
    proj_v = (t_v.T @ src_v) / q_v[:, None]
    return proj_u, proj_v


def normalize_scores(scores, mode):
    if mode == "none":
        return scores
    if mode == "zscore":
        mu = float(scores.mean())
        sigma = float(scores.std())
        return (scores - mu) / max(sigma, 1e-8)
    return scores


def transform_score_space(scores, mode):
    if mode == "sigmoid":
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


def run_for_target(args, target_city):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.source_cities is None:
        source_cities = [c for c in ALL_CITIES if c != target_city]
    else:
        source_cities = args.source_cities

    target_dataset = load_dataset(target_city, args.threshold, args.device)
    tgt_u, tgt_v = load_embeddings(target_city, args.model)
    target_scores = tgt_u @ tgt_v.T
    target_scores = transform_score_space(target_scores, args.score_space)

    base_rec, base_ndcg = get_topk_and_metrics(target_scores, target_dataset, args.topk)
    print(f"Target={target_city}, Model={args.model}")
    print(f"LightGCN baseline: Recall@{args.topk}={base_rec:.4f}, nDCG@{args.topk}={base_ndcg:.4f}")

    infer_scores = []
    infer_sources = []
    for src_city in source_cities:
        src_u, src_v = load_embeddings(src_city, args.model)
        proj_u, proj_v = transport_and_project(src_u, src_v, tgt_u, tgt_v, args.gw_eps)
        s = proj_u @ proj_v.T
        s = transform_score_space(s, args.score_space)
        s = normalize_scores(s, args.score_norm)
        infer_scores.append(s)
        infer_sources.append(src_city)
        print(f"Finished transport: {src_city} -> {target_city}")

    if not infer_scores:
        return {
            "target_city": target_city,
            "baseline_recall": float(base_rec),
            "baseline_ndcg": float(base_ndcg),
            "best_gamma": None,
            "best_recall": float(base_rec),
            "best_ndcg": float(base_ndcg),
            "improv_recall_pct": 0.0,
            "improv_ndcg_pct": 0.0,
            "gamma_results": [],
        }

    gamma_values = [float(x.strip()) for x in args.gammas.split(",") if x.strip()]
    if args.allow_zero_gamma:
        gamma_values = sorted(set([0.0] + gamma_values))

    kept_scores = infer_scores
    kept_sources = infer_sources
    prescreen = []
    if args.prescreen_sources:
        kept_scores, kept_sources = [], []
        for src_city, s in zip(infer_sources, infer_scores):
            local_best = (base_ndcg, base_rec, 0.0)
            for g in gamma_values:
                if g == 0.0:
                    continue
                fused = target_scores + g * s
                rr, nn = get_topk_and_metrics(fused, target_dataset, args.topk)
                if (nn, rr) > (local_best[0], local_best[1]):
                    local_best = (nn, rr, g)
            improv_ndcg = local_best[0] - base_ndcg
            prescreen.append({
                "source_city": src_city,
                "best_single_gamma": float(local_best[2]),
                "best_single_recall": float(local_best[1]),
                "best_single_ndcg": float(local_best[0]),
                "improv_ndcg_abs": float(improv_ndcg),
            })
            if improv_ndcg >= args.min_improv_ndcg and local_best[2] > 0.0:
                kept_scores.append(s)
                kept_sources.append(src_city)
        print(f"Prescreen kept {len(kept_sources)}/{len(infer_sources)} sources for {target_city}: {kept_sources}")
        if not kept_scores:
            kept_scores, kept_sources = infer_scores, infer_sources
            print("No source passed prescreen; fallback to all sources.")

    best = None
    gamma_results = []
    if args.per_source_gamma:
        # Coordinate search for per-source gamma vector.
        if args.allow_zero_gamma:
            gamma_vec = [0.0 for _ in kept_scores]
        else:
            init_gamma = gamma_values[0]
            gamma_vec = [init_gamma for _ in kept_scores]

        def eval_gamma_vec(gv):
            fused = target_scores.copy()
            for ii, ss in enumerate(kept_scores):
                fused += gv[ii] * ss
            rr, nn = get_topk_and_metrics(fused, target_dataset, args.topk)
            return rr, nn

        best_rec, best_ndcg = eval_gamma_vec(gamma_vec)
        for _ in range(args.max_coord_iter):
            improved = False
            for i in range(len(kept_scores)):
                local_best = (best_rec, best_ndcg, gamma_vec[i])
                for g in gamma_values:
                    trial = gamma_vec.copy()
                    trial[i] = g
                    rr, nn = eval_gamma_vec(trial)
                    gamma_results.append({
                        "gamma_vec": trial,
                        "recall": float(rr),
                        "ndcg": float(nn),
                    })
                    if (nn, rr) > (local_best[1], local_best[0]):
                        local_best = (rr, nn, g)
                if local_best[2] != gamma_vec[i]:
                    gamma_vec[i] = local_best[2]
                    best_rec, best_ndcg = local_best[0], local_best[1]
                    improved = True
            if not improved:
                break
        best = {"gamma_vec": gamma_vec, "rec": best_rec, "ndcg": best_ndcg}
        print(
            f"Best OTC-LightGCN per-source gamma={best['gamma_vec']}: "
            f"Recall@{args.topk}={best['rec']:.4f}, nDCG@{args.topk}={best['ndcg']:.4f}"
        )
    else:
        for gamma in gamma_values:
            fused = target_scores.copy()
            for s in kept_scores:
                fused += gamma * s
            rec, ndcg = get_topk_and_metrics(fused, target_dataset, args.topk)
            print(f"gamma={gamma:.1f}: Recall@{args.topk}={rec:.4f}, nDCG@{args.topk}={ndcg:.4f}")
            gamma_results.append({"gamma": float(gamma), "recall": float(rec), "ndcg": float(ndcg)})
            score = (ndcg, rec)
            if best is None or score > (best["ndcg"], best["rec"]):
                best = {"gamma": gamma, "rec": rec, "ndcg": ndcg}
        print(
            f"Best OTC-LightGCN: gamma={best['gamma']:.1f}, "
            f"Recall@{args.topk}={best['rec']:.4f}, nDCG@{args.topk}={best['ndcg']:.4f}"
        )
    return {
        "target_city": target_city,
        "baseline_recall": float(base_rec),
        "baseline_ndcg": float(base_ndcg),
        "best_gamma": float(best["gamma"]) if "gamma" in best else None,
        "best_gamma_vec": best["gamma_vec"] if "gamma_vec" in best else None,
        "kept_sources": kept_sources,
        "source_prescreen": prescreen,
        "best_recall": float(best["rec"]),
        "best_ndcg": float(best["ndcg"]),
        "improv_recall_pct": float((best["rec"] - base_rec) / max(base_rec, 1e-12) * 100.0),
        "improv_ndcg_pct": float((best["ndcg"] - base_ndcg) / max(base_ndcg, 1e-12) * 100.0),
        "gamma_results": gamma_results,
    }


def main():
    global args_global
    args = parse_args()
    args_global = args
    targets = ALL_CITIES if args.all_targets else [args.target_city]
    results = []
    for target in targets:
        results.append(run_for_target(args, target))

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "targets": targets,
            "topk": args.topk,
            "gw_eps": args.gw_eps,
            "gammas": [float(x.strip()) for x in args.gammas.split(",") if x.strip()],
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote OTC metrics to {out_path}")


if __name__ == "__main__":
    main()
