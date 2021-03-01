import numpy as np
import torch
from torch.utils.data import Dataset


def build_quadratic_monoms(point):
    monoms = []
    for i in range(len(point)):
        cur_monoms = point[i:] * point[i]
        monoms.append(cur_monoms)
    return np.concatenate(monoms)


class PointsDataset(Dataset):

    def __init__(self, data):
        self.data = data
        pass

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        q = build_quadratic_monoms(self.data[item, :])
        t = np.concatenate([q, self.data[item, :], [1]]).astype('float32')
        return torch.from_numpy(q.astype('float32')), torch.from_numpy(self.data[item, :].astype('float32'))