import os

import numpy as np
import torch
import torch.nn.functional as F

from preprocess import load_sparse


class EHRDataset:
    def __init__(self, data_path, label='d', batch_size=32, shuffle=True, device=torch.device('cpu')):
        super().__init__()
        self.task = label
        self.path = data_path
        self.code_x, self.pro_x, self.visit_lens, self.y, self.adm_time, self.dis_time = self._load()
        self._size = self.code_x.shape[0]
        self.idx = np.arange(self._size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _load(self):
        code_x = load_sparse(os.path.join(self.path, 'code_x.npz'))
        visit_lens = load_sparse(os.path.join(self.path, 'visit_lens.npz'))
        y = load_sparse(os.path.join(self.path, 'y.npz'))
        pro_x = load_sparse(os.path.join(self.path, 'proc_x.npz'))
        adm_time = load_sparse(os.path.join(self.path, 'adm_time.npz'))
        dis_time = load_sparse(os.path.join(self.path, 'dis_time.npz'))
        return code_x, pro_x, visit_lens, y, adm_time, dis_time

    def on_epoch_end(self):
        # 每个epoch结束执行该函数
        if self.shuffle:
            np.random.shuffle(self.idx)

    def size(self):
        return self._size

    def label(self):
        return self.y

    def __len__(self):
        len_ = self._size // self.batch_size
        return len_ if self._size % self.batch_size == 0 else len_ + 1

    def __getitem__(self, index):
        device = self.device
        start = index * self.batch_size
        end = start + self.batch_size
        slices = self.idx[start:end]
        code_x = torch.from_numpy(self.code_x[slices]).to(device)
        visit_lens = torch.from_numpy(self.visit_lens[slices]).to(device=device, dtype=torch.long)
        pro_x = torch.from_numpy(self.pro_x[slices]).to(device)
        y = torch.from_numpy(self.y[slices]).to(device=device, dtype=torch.float32)
        adm_time = torch.from_numpy(self.adm_time[slices]).to(device=device, dtype=torch.float32)
        dis_time = torch.from_numpy(self.dis_time[slices]).to(device=device, dtype=torch.float32)
        return code_x, pro_x, visit_lens, y, adm_time, dis_time



def format_time(seconds):
    if seconds <= 60:
        time_str = '%.1fs' % seconds
    elif seconds <= 3600:
        time_str = '%dm%.1fs' % (seconds // 60, seconds % 60)
    else:
        time_str = '%dh%dm%.1fs' % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)
    return time_str


def medical_codes_loss(y_pred, y_true):
    per_sample_losses = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    summed_losses = per_sample_losses.sum(dim=-1)
    mean_loss = summed_losses.mean()
    return mean_loss


class MultiStepLRScheduler:
    def __init__(self, optimizer, epochs, init_lr, milestones, lrs):
        self.optimizer = optimizer
        self.epochs = epochs
        self.init_lr = init_lr
        self.lrs = self._generate_lr(milestones, lrs)
        self.current_epoch = 0

    def _generate_lr(self, milestones, lrs):
        milestones = [1] + milestones + [self.epochs + 1]
        lrs = [self.init_lr] + lrs
        lr_grouped = np.concatenate([np.ones((milestones[i + 1] - milestones[i],)) * lrs[i]
                                     for i in range(len(milestones) - 1)])
        return lr_grouped

    def step(self):
        lr = self.lrs[self.current_epoch]
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        self.current_epoch = 0
