import numpy as np
import pandas as pd
import os
import json
import torch
import argparse
import time
from pathlib import Path
from data_utils import OpenSiteRec, split
from eval_utils import PrecisionRecall_atK, NDCG_atK, get_label
from model import VanillaMF, NeuMF, RankNet, BasicCTRModel, WideDeep, DeepFM, xDeepFM, NGCF, LightGCN


MODEL = {'VanillaMF': VanillaMF, 'NeuMF': NeuMF, 'RankNet': RankNet,
         'DNN': BasicCTRModel, 'WideDeep': WideDeep, 'DeepFM': DeepFM, 'xDeepFM': xDeepFM,
         'NGCF': NGCF, 'LightGCN': LightGCN}


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': -1,
        'epochs': 300,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'LightGCN',
        'dim': 100,
        'layers': 2,
        'city': 'Singapore',
        'threshold': 5,
        'topk': [20],
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 10,
        'batch_size': 128,
        'save': 0,
        'metrics_out': '',
        'force_split': 0,
        'split_seed': -1,
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        parser.add_argument(f"--{param}", default=val)
    args = parser.parse_args()
    args.lr = float(args.lr)
    args.dropout = float(args.dropout)
    args.cuda = int(args.cuda)
    args.epochs = int(args.epochs)
    args.weight_decay = float(args.weight_decay)
    args.seed = int(args.seed)
    args.dim = int(args.dim)
    args.layers = int(args.layers)
    args.threshold = int(args.threshold)
    if isinstance(args.topk, str):
        args.topk = [int(x.strip()) for x in args.topk.strip("[]").split(",") if x.strip()]
    elif isinstance(args.topk, int):
        args.topk = [args.topk]
    args.patience = int(args.patience)
    args.eval_freq = int(args.eval_freq)
    args.lr_reduce_freq = int(args.lr_reduce_freq)
    args.batch_size = int(args.batch_size)
    args.save = int(args.save)
    args.force_split = int(args.force_split)
    args.split_seed = int(args.split_seed)
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

split_seed = args.seed if args.split_seed < 0 else args.split_seed
split(args.city, args.threshold, force=bool(args.force_split), seed=split_seed)
dataset = OpenSiteRec(args)
print(dataset.testDataSize)
args.user_num, args.item_num, args.cate_num = dataset.n_user, dataset.m_item, dataset.k_cate
args.Graph = dataset.Graph
model = MODEL[args.model](args)
print(str(model))
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')


def build_eval_dict(df):
    out = {}
    for _, row in df.iterrows():
        user, item = int(row['Brand_ID']), int(row['Region_ID'])
        out[user] = out.get(user, [])
        out[user].append(item)
    # Deduplicate regions per brand so each relevant region is counted once.
    out = {u: sorted(set(items)) for u, items in out.items()}
    return out


val_dict = build_eval_dict(dataset.val_data)


def train():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        users, labels = torch.LongTensor(dataset.U[indices]).to(args.device), \
                        torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(users)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_graph():
    model.train()
    model.mode = 'train'
    dataset.uniform_sampling()
    batch_num = dataset.trainDataSize // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.trainDataSize \
            else torch.arange(i * args.batch_size, dataset.trainDataSize)
        batch = dataset.S[indices]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
                                      torch.LongTensor(batch[:, 1]).to(args.device), \
                                      torch.LongTensor(batch[:, 2]).to(args.device)

        loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
        loss = loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_CTR():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        instances = {'Brand_ID': torch.LongTensor(dataset.U[indices]).to(args.device),
                     'Cate1_ID': torch.LongTensor(dataset.bF[indices][:, 0]).to(args.device),
                     'Cate2_ID': torch.LongTensor(dataset.bF[indices][:, 1]).to(args.device),
                     'Cate3_ID': torch.LongTensor(dataset.bF[indices][:, 2]).to(args.device)}
        labels = torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(instances)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def evaluate(eval_dict):
    model.eval()
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        model.mode = 'test'
    all_pos = dataset.allPos
    rec, ndcg = 0., 0.
    with torch.no_grad():
        users = list(eval_dict.keys())
        items = [eval_dict[u] for u in users]
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i * args.batch_size: (i + 1) * args.batch_size] \
                if (i + 1) * args.batch_size <= len(users) else users[i * args.batch_size:]
            batch_pos = [all_pos[u] for u in batch_users]
            batch_items = [items[u] for u in batch_users]
            if args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
                instances = {'Brand_ID': torch.LongTensor(dataset.U[batch_users]).to(args.device),
                         'Cate1_ID': torch.LongTensor(dataset.F[batch_users][:, 0]).to(args.device),
                         'Cate2_ID': torch.LongTensor(dataset.F[batch_users][:, 1]).to(args.device),
                         'Cate3_ID': torch.LongTensor(dataset.F[batch_users][:, 2]).to(args.device)}
            else:
                instances = torch.LongTensor(batch_users).to(args.device)

            ratings = model(instances)
            for range_i, its in enumerate(batch_pos):
                ratings[range_i, its] = -1e10
            _, ratings_K = torch.topk(ratings, k=args.topk[-1])
            ratings_K = ratings_K.cpu().numpy()

            r = get_label(batch_items, ratings_K)
            for k in args.topk:
                _, batch_rec = PrecisionRecall_atK(batch_items, r, k)
                batch_ndcg = NDCG_atK(batch_items, r, k)
                rec += batch_rec * len(batch_users)
                ndcg += batch_ndcg * len(batch_users)

        rec /= len(users)
        ndcg /= len(users)
    return rec, ndcg


def export_embeddings():
    if args.model == 'LightGCN':
        model.mode = 'test'
        with torch.no_grad():
            user_emb, item_emb = model._LightGCN__message_passing()
        return user_emb.detach().cpu(), item_emb.detach().cpu()
    if args.model == 'NGCF':
        model.mode = 'test'
        with torch.no_grad():
            user_emb, item_emb = model._NGCF__message_passing()
        return user_emb.detach().cpu(), item_emb.detach().cpu()
    return model.user_embedding.weight.data.detach().cpu(), model.item_embedding.weight.data.detach().cpu()


t_total = time.time()
best_val_ndcg, best_epoch = -1., -1
best_state = None
wait = 0
for epoch in range(args.epochs):
    if args.model in ['VanillaMF', 'RankNet', 'NGCF', 'LightGCN']:
        train_graph()
    elif args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
        train_CTR()
    else:
        train()
    torch.cuda.empty_cache()
    if (epoch + 1) % args.eval_freq == 0 or epoch == 0:
        val_rec, val_ndcg = evaluate(val_dict)
        test_rec, test_ndcg = evaluate(dataset.testDict)
        print(f'Epoch {epoch} | Val Recall@{args.topk[-1]}: {val_rec:.4f} | Val nDCG@{args.topk[-1]}: {val_ndcg:.4f}')
        print(f'Epoch {epoch} | Test Recall@{args.topk[-1]}: {test_rec:.4f} | Test nDCG@{args.topk[-1]}: {test_ndcg:.4f}')
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}, best epoch={best_epoch}")
                break
        torch.cuda.empty_cache()

if best_state is not None:
    model.load_state_dict(best_state)

final_rec, final_ndcg = evaluate(dataset.testDict)
elapsed_sec = time.time() - t_total
print(f'Best Epoch: {best_epoch}')
print(f'Best Results: \nRecall@{args.topk[-1]}: {round(final_rec, 4)}\nnDCG@{args.topk[-1]}: {round(final_ndcg, 4)}')
print(f"Elapsed Time (s): {elapsed_sec:.2f}")

if args.save:
    save_dir = Path(__file__).resolve().parent.parent / args.city / "split"
    os.makedirs(save_dir, exist_ok=True)
    final_user_emb, final_item_emb = export_embeddings()
    torch.save(final_user_emb, save_dir / f"{args.model}_U_emb.pt")
    torch.save(final_item_emb, save_dir / f"{args.model}_V_emb.pt")
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_rec': final_rec,
        'best_ndcg': final_ndcg,
        'elapsed_sec': elapsed_sec,
    }, save_dir / f"{args.model}_checkpoint.pth")
    print(f'Saved embeddings to {save_dir}')

if args.metrics_out:
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "city": args.city,
        "model": args.model,
        "best_epoch": int(best_epoch),
        "recall_at_20": float(final_rec),
        "ndcg_at_20": float(final_ndcg),
        "elapsed_sec": float(elapsed_sec),
        "epochs": int(args.epochs),
        "eval_freq": int(args.eval_freq),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
