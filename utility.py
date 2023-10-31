import os
import random
import numpy as np
import scipy.sparse as sp
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BundleTrainDataset(Dataset):
    def __init__(self, conf, b_i_pairs, b_i_graph, features, num_bundles, b_i_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.b_i_pairs = b_i_pairs
        self.b_i_graph = b_i_graph
        self.bundles_map = np.argwhere(self.b_i_graph.sum(axis=1) > 0)[
            :, 0].reshape(-1)
        self.num_bundles = num_bundles
        self.num_items = self.b_i_graph.shape[1]
        self.neg_sample = neg_sample
        self.features = features

        self.b_i_for_neg_sample = b_i_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample

        self.len_max = int(self.b_i_graph.sum(axis=1).max())
        if self.conf["bundle_ratio"] > 1:
            self.num_add = round(
                self.len_max * (self.conf["bundle_ratio"] - 1))
            self.num_add = self.num_add if self.num_add > 0 else 1
            self.len_max = self.len_max + self.num_add

        if self.len_max > self.conf["num_token"]:
            self.len_max = self.conf["num_token"]
        print(f"Train: {self.len_max}")

        self.bundle_augment = conf["bundle_augment"]

    def __getitem__(self, index):

        full = torch.from_numpy(
            self.b_i_graph[self.bundles_map[index]].toarray()).squeeze()

        # multi-hot
        modify = torch.zeros_like(full)
        indices = torch.argwhere(full)[:, 0]

        # shuffle >>>
        num_items = indices.shape[0]
        random_idx = torch.randperm(num_items)
        indices = indices[random_idx]
        # shuffle <<<

        seq_full = F.pad(
            indices, (0, self.len_max-len(indices)), value=self.num_items)

        if self.conf["bundle_ratio"] > 0 and self.conf["bundle_ratio"] < 1:  # remove items
            if self.bundle_augment == "ID":
                line = round(len(indices)*self.conf["bundle_ratio"]+0.5)
                line = line if line < len(indices) else len(
                    indices)-1  # ensure at less one item is masked
                p_indices = indices[:line]
                modify[p_indices] = 1

                # sequence set:
                seq_modify = F.pad(
                    p_indices, (0, self.len_max-len(p_indices)), value=self.num_items)
            elif self.bundle_augment == "IR":
                line = round(len(indices)*self.conf["bundle_ratio"]+0.5)
                line = line if line < len(indices) else len(
                    indices)-1  # ensure at less one item is masked
                p_indices = indices[:line]
                replace_indices = random.sample(
                    range(num_items), len(indices) - line)
                replace_indices = torch.LongTensor(
                    replace_indices).to(p_indices.device)
                p_indices = torch.cat([p_indices, replace_indices])

                modify[p_indices] = 1
                seq_modify = F.pad(
                    p_indices, (0, self.len_max-len(p_indices)), value=self.num_items)

        elif self.conf["bundle_ratio"] == 1:
            modify = full
            seq_modify = seq_full
        elif self.conf["bundle_ratio"] > 1:  # add items
            m_indices = indices.tolist()
            # randomly add items
            while True:
                i = np.random.randint(self.num_items)
                if not i in m_indices:
                    m_indices.append(i)
                    if len(m_indices) == self.num_add+len(indices):
                        break

            m_indices = torch.LongTensor(m_indices)
            modify[m_indices] = 1
            seq_modify = F.pad(
                m_indices, (0, self.len_max-len(m_indices)), value=self.num_items)

        return self.bundles_map[index], full, seq_full, modify, seq_modify

    def __len__(self):
        return len(self.bundles_map)


class BundleTestDataset(Dataset):
    def __init__(self, conf, b_i_pairs_i, b_i_graph_i, b_i_pairs_gt, b_i_graph_gt, num_bundles, num_items):
        self.b_i_pairs_i = b_i_pairs_i
        self.b_i_graph_i = b_i_graph_i

        self.bundles_map = np.argwhere(self.b_i_graph_i.sum(axis=1) > 0)[
            :, 0].reshape(-1)
        self.b_i_pairs_gt = b_i_pairs_gt
        self.b_i_graph_gt = b_i_graph_gt

        self.num_bundles = num_bundles
        self.num_items = num_items

        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        if self.len_max > conf["num_token"]:
            self.len_max = conf["num_token"]
        print(f"Val/Test: {self.len_max}")

    def __getitem__(self, index):
        graph_index = self.bundles_map[index]
        b_i_i = torch.from_numpy(
            self.b_i_graph_i[graph_index].toarray()).squeeze()
        b_i_gt = torch.from_numpy(
            self.b_i_graph_gt[graph_index].toarray()).squeeze()

        indices = torch.argwhere(b_i_i)[:, 0]
        seq_b_i_i = F.pad(
            indices, (0, self.len_max-len(indices)), value=self.num_items)

        return self.bundles_map[index], b_i_i, seq_b_i_i, b_i_gt

    def __len__(self):
        return len(self.bundles_map)


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        self.device = conf["device"]
        self.is_openai_embedding = conf["is_openai_embedding"] if "is_openai_embedding" in conf else False
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        u_i_pairs, u_i_graph = self.get_ui()

        b_i_pairs_train, b_i_graph_train = self.get_bi_train()
        b_i_pairs_val_i, b_i_graph_val_i, b_i_pairs_val_gt, b_i_graph_val_gt = self.get_bi(
            "valid")
        b_i_pairs_test_i, b_i_graph_test_i, b_i_pairs_test_gt, b_i_graph_test_gt = self.get_bi(
            "test")

        b_i_for_neg_sample, b_b_for_neg_sample = None, None

        b_i_pairs_seen, b_i_graph_seen = self.combine_graph(
            [b_i_pairs_train, b_i_pairs_val_i, b_i_pairs_test_i],
            shape=(self.num_bundles, self.num_items),
            tag="BI(seen)")
        self.graphs = [u_i_graph, b_i_graph_train, b_i_graph_seen]

        self.features = self.get_features()

        self.bundle_train_data = BundleTrainDataset(
            conf, b_i_pairs_train, b_i_graph_train, self.features, self.num_bundles, b_i_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])

        self.bundle_val_data = BundleTestDataset(conf, b_i_pairs_val_i, b_i_graph_val_i, b_i_pairs_val_gt, b_i_graph_val_gt,
                                                 self.num_bundles, self.num_items)
        self.bundle_test_data = BundleTestDataset(conf, b_i_pairs_test_i, b_i_graph_test_i, b_i_pairs_test_gt, b_i_graph_test_gt,
                                                  self.num_bundles, self.num_items)

        self.train_loader = DataLoader(
            self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=10)
        self.val_loader = DataLoader(
            self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=20)
        self.test_loader = DataLoader(
            self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=20)

    def combine_graph(self, pairs_list, shape, tag):
        pairs = np.concatenate(pairs_list, axis=0)
        indice = np.array(pairs, dtype=np.int32)
        values = np.ones(len(pairs), dtype=np.float32)
        graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=shape)
        return pairs, graph

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, 'count.json'), 'r') as f:
            self.stat = json.loads(f.read())
        return self.stat["#U"], self.stat["#B"], self.stat["#I"]

    def get_features(self):
        try:
            content_feature = torch.load(os.path.join(
                self.path, self.name, 'content_feature.pt'), map_location=self.device)
            if not self.is_openai_embedding:
                description_feature = torch.load(os.path.join(
                    self.path, self.name, 'description_feature.pt'), map_location=self.device)
            else:
                description_feature = torch.load(os.path.join(
                    self.path, self.name, 'openai_description_feature.pt'), map_location=self.device)
        except:
            print("[ERROR] no content_feature & description_feature")
            content_feature = description_feature = None

        cf_feature = torch.load(os.path.join(
            self.path, self.name, 'item_cf_feature.pt'), map_location=self.device)
        return (content_feature, description_feature, cf_feature)

    def get_ui(self):
        u_i_pairs = np.load(os.path.join(self.path, self.name, 'ui_full.npy'))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items))

        return u_i_pairs, u_i_graph

    def get_bi_train(self):

        b_i_pairs = np.load(os.path.join(self.path, self.name, 'bi_train.npy'))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.csr_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items))

        return b_i_pairs, b_i_graph

    def get_bi(self, task):

        b_i_pairs_i = np.load(os.path.join(
            self.path, self.name, f'bi_{task}_input.npy'))
        b_i_pairs_gt = np.load(os.path.join(
            self.path, self.name, f'bi_{task}_gt.npy'))

        b_i_graph_i = pairs2csr(
            b_i_pairs_i, (self.num_bundles, self.num_items))
        b_i_graph_gt = pairs2csr(
            b_i_pairs_gt, (self.num_bundles, self.num_items))

        return b_i_pairs_i, b_i_graph_i, b_i_pairs_gt, b_i_graph_gt


def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape)
