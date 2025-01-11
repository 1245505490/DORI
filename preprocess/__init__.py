import os
import numpy as np

def save_sparse(path, x):
    idx = np.where(x > 0)
    values = x[idx]
    np.savez(path, idx=idx, values=values, shape=x.shape)

def load_sparse(path):
    data = np.load(path, allow_pickle=True)
    idx, values = data['idx'], data['values']
    mat = np.zeros(data['shape'], dtype=values.dtype)
    mat[tuple(idx)] = values
    return mat

def save_data(path, code_x, proc_x, visit_lens, y, adm_time, dis_time):
    save_sparse(os.path.join(path, 'code_x'), code_x)
    save_sparse(os.path.join(path, 'visit_lens'), visit_lens)
    save_sparse(os.path.join(path, 'y'), y)
    save_sparse(os.path.join(path, 'proc_x'), proc_x)
    save_sparse(os.path.join(path, 'adm_time'), adm_time)
    save_sparse(os.path.join(path, 'dis_time'), dis_time)
