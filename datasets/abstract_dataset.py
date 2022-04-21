import random
from abc import ABC, abstractmethod
from os import makedirs, path

import numpy as np
import torch
import torch.utils.data

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

project_root = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))


class AbstractDataset(ABC, torch.utils.data.Dataset):

    @abstractmethod
    def __init__(self, name, split, p_test, p_val):
        if split not in ['train', 'test', 'validation']:
            raise ValueError('Unknown dataset split')

        self.split = split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = path.join(project_root, 'data', name)
        self.p_test = p_test
        self.p_val = p_val
        makedirs(self.data_dir, exist_ok=True)

    def _discretize_continuous(self, features, continuous, categorical, rem,
                               all_bins=None, k=4, column_ids=None):
        new_features = []
        new_categorical = []
        new_bins = []

        beg = 0
        new_column_ids = dict() if column_ids is not None else None

        for idx, i in enumerate(continuous):
            q = np.linspace(0, 1, k+1)
            if all_bins is None:
                bins = np.quantile(features[:, i].cpu().numpy(), q)
                bins[0], bins[-1] = bins[0] - 1e-2, bins[-1] + 1e-2
                new_bins += [bins]
            else:
                bins = all_bins[idx]

            if column_ids is not None:
                for col, col_id in column_ids.items():
                    if col_id != i:
                        continue

                    col_to_remove = col

                    for j in range(k):
                        lb, ub = bins[j], bins[j + 1]
                        new_column_ids[f'{col}={lb}-{ub}'] = beg + j

                column_ids.pop(col_to_remove)

            disc = np.digitize(features[:, i].cpu().numpy(), bins)
            disc = np.clip(disc, 1, k) # test data outside of training bins
            assert np.all(disc >= 1)
            assert np.all(disc <= k)
            one_hot_vals = torch.zeros(features.shape[0], k).to(features.device)
            one_hot_vals[np.arange(features.shape[0]), disc - 1] = 1.0
            new_features += [one_hot_vals]
            new_categorical += [[j for j in range(beg, beg+k)]]
            beg += k

        for col_name, col_ids in categorical.items():
            if col_name in rem:
                if column_ids is not None:
                    cols_to_remove = list(
                        filter(
                            lambda c: c.startswith(col_name), column_ids.keys()
                        )
                    )
                    for col_to_remove in cols_to_remove:
                        column_ids.pop(col_to_remove)
                continue

            if column_ids is not None:
                for col, col_id in column_ids.items():
                    if col.startswith(col_name):
                        new_column_ids[col] = beg + col_id - col_ids[0]
                cols_to_remove = list(
                    filter(lambda c: c.startswith(col_name), column_ids.keys())
                )
                for col_to_remove in cols_to_remove:
                    column_ids.pop(col_to_remove)

            new_features += [features[:, col_ids]]
            new_categorical += [[j for j in range(beg, beg+len(col_ids))]]
            beg += len(col_ids)
        new_features = torch.cat(new_features, dim=1)

        return new_features, new_categorical, new_bins, new_column_ids

        
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.protected[index]

    def __len__(self):
        return self.labels.size()[0]

    def _normalize(self, columns):
        columns = columns if columns is not None else np.arange(self.X_train.shape[1])

        self.mean, self.std = self.X_train.mean(dim=0)[columns], self.X_train.std(dim=0)[columns]

        self.X_train[:, columns] = (self.X_train[:, columns] - self.mean) / self.std
        self.X_val[:, columns] = (self.X_val[:, columns] - self.mean) / self.std
        self.X_test[:, columns] = (self.X_test[:, columns] - self.mean) / self.std

    def _assign_split(self):
        if self.split == 'train':
            self.features, self.labels, self.protected = self.X_train, self.y_train, self.protected_train
        elif self.split == 'test':
            self.features, self.labels, self.protected = self.X_test, self.y_test, self.protected_test
        elif self.split == 'validation':
            self.features, self.labels, self.protected = self.X_val, self.y_val, self.protected_val

        self.features = self.features.float()
        self.labels = self.labels.float()
        self.protected = self.protected.long()

    def pos_weight(self, split):
        if split == 'train':
            labels = self.y_train
        elif split == 'train-val':
            labels = torch.cat((self.y_train, self.y_val))
        else:
            raise ValueError('Unknown split')

        positives = torch.sum(labels == 1).float()
        negatives = torch.sum(labels == 0).float()

        assert positives + negatives == labels.shape[0]

        return negatives / positives
