import torch
from .read_vecs import READER

class VectorDataset:
    def __init__(self, dataset='sift1m', data_path=None, normalize=True, train_size=10**5, test_size=10**6, device='cpu'):
        self.dataset = dataset
        self.data_path = data_path
        self.normalize = normalize

        xt, xb, xq, gt = READER[dataset](data_path, train_size, test_size)
        
        self.xt = torch.from_numpy(xt).to(device)
        self.xb = torch.from_numpy(xb).to(device)
        self.xq = torch.from_numpy(xq).to(device)
        self.gt = gt

        if normalize:
            mean_norm = self.xt.norm(p=2, dim=-1).mean().item()
            self.xt = self.xt / mean_norm
            self.xb = self.xb / mean_norm
            self.xq = self.xq / mean_norm
            