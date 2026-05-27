import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from pathlib import Path


BASE_DATA_DIR = Path(__file__).resolve().parent.parent


def _city_dir(city):
    return BASE_DATA_DIR / city


def _split_dir(city):
    return _city_dir(city) / "split"


def split(city='NYC', threshold=20, force=False, seed=42):
    split_dir = _split_dir(city)
    train_file = split_dir / "train.pkl"
    val_file = split_dir / "val.pkl"
    test_file = split_dir / "test.pkl"

    if not force and train_file.exists() and val_file.exists() and test_file.exists():
        print(f"Existing split detected for {city}; skip splitting.")
        return

    split_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(_city_dir(city) / f"{city}_KG_plus.pkl")
    while True:
        brand_counts = df['Brand'].value_counts()
        valid_brands = brand_counts[brand_counts >= threshold].index
        if len(valid_brands) == len(brand_counts):
            break 
        df = df[df['Brand'].isin(valid_brands)]
        
    
    df.reset_index(inplace=True, drop=True)
    print(f"After {threshold}-core: {len(df)} POIs, {df['Brand'].nunique()} brands")

    brand2id, cate12id, cate22id, cate32id = {}, {}, {}, {}
    for idx, row in df.iterrows():
        brand, cate_1, cate_2, cate_3 = row['Brand'], row['cate_1'], row['cate_2'], row['cate_3']
        if brand not in brand2id.keys():
            brand2id[brand] = len(brand2id)
        if cate_1 not in cate12id.keys():
            cate12id[cate_1] = len(cate12id)
        if cate_2 not in cate22id.keys():
            cate22id[cate_2] = len(cate22id)
        if cate_3 not in cate32id.keys():
            cate32id[cate_3] = len(cate32id)

    brand2id = pd.DataFrame({'Brand': list(brand2id.keys()), 'Brand_ID': list(brand2id.values())})
    cate12id = pd.DataFrame({'cate_1': list(cate12id.keys()), 'Cate1_ID': list(cate12id.values())})
    cate22id = pd.DataFrame({'cate_2': list(cate22id.keys()), 'Cate2_ID': list(cate22id.values())})
    cate32id = pd.DataFrame({'cate_3': list(cate32id.keys()), 'Cate3_ID': list(cate32id.values())})

    df = df.merge(brand2id, on=['Brand'], how='left')
    df = df.merge(cate12id, on=['cate_1'], how='left')
    df = df.merge(cate22id, on=['cate_2'], how='left')
    df = df.merge(cate32id, on=['cate_3'], how='left')
    df = df[['ID', 'Name', 'Brand_ID', 'Cate1_ID', 'Cate2_ID', 'Cate3_ID', 'Region_ID']]

    print(df['Brand_ID'].max())
    print(df['Region_ID'].max())

    np.random.seed(seed)
    train_data, val_data, test_data = [], [], []
    for i in range(df['Brand_ID'].max() + 1):
        data = df[df['Brand_ID'] == i] 
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            data[['Brand_ID', 'Cate1_ID', 'Cate2_ID', 'Cate3_ID']], data['Region_ID'],
            test_size=0.2, random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val, y_train_val, test_size=0.125, random_state=seed)
        x_train['Region_ID'] = y_train
        x_val['Region_ID'] = y_val
        x_test['Region_ID'] = y_test
        train_data.append(x_train)
        val_data.append(x_val)
        test_data.append(x_test)

    train_data, val_data, test_data = pd.concat(train_data, axis=0), pd.concat(val_data, axis=0), pd.concat(test_data, axis=0)
    train_data.to_pickle(train_file)
    val_data.to_pickle(val_file)
    test_data.to_pickle(test_file)


class OpenSiteRec(Dataset):
    def __init__(self, args):
        super(OpenSiteRec, self).__init__()
        self.device = args.device
        self.city = args.city
        split_dir = _split_dir(args.city)
        self.train_data = pd.read_pickle(split_dir / "train.pkl")
        self.val_data = pd.read_pickle(split_dir / "val.pkl")
        self.test_data = pd.read_pickle(split_dir / "test.pkl")
        self.n_user = int(max(self.train_data['Brand_ID'].max(), self.test_data['Brand_ID'].max()) + 1)
        self.m_item = int(max(self.train_data['Region_ID'].max(), self.test_data['Region_ID'].max()) + 1)
        self.k_cate = [int(max(self.train_data['Cate1_ID'].max(), self.test_data['Cate1_ID'].max()) + 1),
                       int(max(self.train_data['Cate2_ID'].max(), self.test_data['Cate2_ID'].max()) + 1),
                       int(max(self.train_data['Cate3_ID'].max(), self.test_data['Cate3_ID'].max()) + 1)]
        self.trainDataSize, self.testDataSize = self.train_data.shape[0], self.test_data.shape[0]
        self.UserItemNet = csr_matrix((np.ones(self.trainDataSize),
                                       (self.train_data['Brand_ID'], self.train_data['Region_ID'])),
                                      shape=(self.n_user, self.m_item))
        self.allPos = self.get_user_pos_items(list(range(self.n_user)))
        self.U, self.F, self.I = np.array(list(range(self.n_user))), [], []
        for user in range(self.n_user):
            features = self.train_data[self.train_data['Brand_ID'] == user]
            self.F.append([features['Cate1_ID'].value_counts().index.tolist()[0],
                           features['Cate2_ID'].value_counts().index.tolist()[0],
                           features['Cate3_ID'].value_counts().index.tolist()[0]])
            user_pos = self.allPos[user]
            user_label = torch.zeros(self.m_item, dtype=torch.float)
            user_label[user_pos] = 1.
            user_label = 0.9 * user_label + (1.0 / self.m_item)
            self.I.append(user_label.tolist())
        self.F = torch.LongTensor(self.F)
        self.I = torch.FloatTensor(self.I)
        self.bF, self.bI = None, None
        item_counts = np.sum(np.array(self.I), axis=0)
        lt_threshold = sorted(item_counts)[int(len(item_counts) * 0.9)]
        self.lt_mask = (item_counts < lt_threshold).astype(float)
        self.testDict = self.__build_test()
        self.Graph = None
        self.get_sparse_graph()
        self.S = None
        self.uniform_sampling()

    def init_batches(self):
        np.random.shuffle(self.U)
        self.bF = self.F[self.U]
        self.bI = self.I[self.U]

    def uniform_sampling(self):
        users = np.random.randint(0, self.n_user, self.trainDataSize)
        allPos = self.allPos
        S = []
        for i, user in enumerate(users):
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, self.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
        self.S = torch.LongTensor(S)

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def __build_test(self):
        td = {}
        for idx, row in self.test_data.iterrows():
            user, item = row[0], row[-1]
            td[user] = td.get(user, [])
            td[user].append(item)
            # if self.lt_mask[item] > 0:
            #     td[user].append(item)
        td = {u: sorted(set(items)) for u, items in td.items()}
        return td

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float64)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading matrix")
        if self.Graph is None:
            print("generating adjacency matrix")
            adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float64)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            print(f"saved norm_mat...")
            try:
                sp.save_npz(_split_dir(self.city) / "s_pre_adj_mat.npz", norm_adj)
            except PermissionError:
                print(f"Warning: cannot write s_pre_adj_mat.npz for {self.city}; continue without saving cache.")

            self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(self.device)

        return self.Graph

    def __getitem__(self, idx):
        return self.S[idx]

    def __len__(self):
        return len(self.S)
