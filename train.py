import _pickle as pickle
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from metrics import evaluate_codes
from models.model import Model
from utils import EHRDataset, format_time, medical_codes_loss, MultiStepLRScheduler


def load_data(encoded_path):
    code_map = pickle.load(open(os.path.join(encoded_path, 'code_map.pkl'), 'rb'))
    pro_code_map = pickle.load(open(os.path.join(encoded_path, 'pro_code_map.pkl'), 'rb'))
    return code_map, pro_code_map


def getLeaf(code_levels, device):
    leaf_list = [torch.full((len(level),), level[-1], dtype=torch.long, device=device) for level in code_levels]
    leaf_list = torch.stack(leaf_list)
    ancestor_list = torch.tensor(code_levels, dtype=torch.long, device=device)
    return leaf_list, ancestor_list


def getCooccur(code_adj, device):
    code_num = len(code_adj) - 1
    max_len = int(torch.max(torch.sum((code_adj > 0).to(torch.int), dim=-1)))
    leaf = torch.zeros(code_num + 1, max_len, dtype=torch.long, device=device)
    mask = torch.zeros_like(leaf, dtype=torch.float, device=device)
    beta = torch.zeros_like(leaf, dtype=torch.float, device=device)
    for i in range(1, code_num + 1):
        indices = torch.nonzero(code_adj[i, :], as_tuple=False).squeeze()
        if indices.dim() > 0:
            leaf[i, :indices.size(0)] = indices
            mask[i, :indices.size(0)] = 1.0
            beta[i, :indices.size(0)] = code_adj[i, indices]

    beta += (beta == 0.) * -1e10
    beta = torch.softmax(beta, dim=1)
    return leaf, mask, beta


if __name__ == '__main__':
    use_cuda = True
    use_proc = False
    if use_proc:
        print("use proc")
    else:
        print("Not use proc")
    dataset = 'mimic3'
    if use_proc:
        data_path = os.path.join('data', dataset, 'proc')
    else:
        data_path = os.path.join('data', dataset, 'diag')
    encoded_path = os.path.join(data_path, 'encoded')
    standard_path = os.path.join(data_path, 'standard')
    train_path = os.path.join(standard_path, 'train')
    valid_path = os.path.join(standard_path, 'valid')
    test_path = os.path.join(standard_path, 'test')
    seed = 18
    task = 'd'  # d,m,k,h
    print(f"dataset: {dataset}")

    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    print(f"device:{device}")
    batch_size = 32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    code_map, pro_code_map = load_data(encoded_path)
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    code_adj, pro_code_adj, code_levels = auxiliary['code_adj'], auxiliary['pro_code_adj'], auxiliary['code_levels']
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, device=device, batch_size=batch_size, shuffle=True)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, device=device, batch_size=batch_size, shuffle=False)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, device=device, batch_size=batch_size, shuffle=False)
    code_num = len(code_map) - 1
    if task == 'd':
        code_dim = 64
    else:
        code_dim = 8
    code_dims = [code_dim] * 4

    leaf_list, ancestor_list = getLeaf(code_levels, device)
    code_adj = torch.from_numpy(code_adj).to(device)
    pro_code_adj = torch.from_numpy(pro_code_adj).to(device)
    leaf, mask, beta = getCooccur(pro_code_adj, device)

    disease_dict = pickle.load(open(os.path.join(data_path, 'disease_emb.pkl'), 'rb'))
    list = []
    for i in range(len(disease_dict)):
        list.append(disease_dict[i])
    disease_emb = torch.stack(list, dim=0)

    config = {
        'use_proc': use_proc,
        'disease_emb': disease_emb.to(dtype=torch.float32, device=device),
        'leaf_list': leaf_list,
        'ancestor_list': ancestor_list,
        'leaf': leaf,
        'beta': beta,
        'leaf_mask': mask,
        'code_adj': pro_code_adj,
        'code_levels': torch.tensor(code_levels[1:], dtype=torch.int32, device=device),
        'code_num_in_levels': np.max(code_levels, axis=0),
        'code_num': code_num,
        'vocab_size': code_num + 1,
        'max_visit_seq_len': train_data.code_x.shape[1],
        'lambda': 0.3,
        'device': device,
    }

    hyper_params = {
        'word_dim': disease_emb.size(-1),
        'code_dims': code_dims,
        'gcn_hidden': code_dim * 4,  # 256
        'gcn_out': code_dim * 2,  # 128
        'gat_hidden': 32,
        'hidden_dim': code_dim * 2,  # 128
        'intermediate_dims': code_dim * 8,  # 512
        'quiry_dim': 32,
        'time_dim': 32,
        'attention_dim': 32,
        'num_layers': 2,
        'num_heads': 4,
        'ffn_dim': 1024,
        'gru_dims': [64],
        'dropout': 0.3
    }


    def lr_schedule_fn(epoch):
        if epoch < 20:
            return 0.1
        elif epoch < 100:
            return 0.05
        elif epoch < 200:
            return 0.025
        else:
            return 0.001


    task_conf = {
        'd': {
            'output_size': code_num,
            'epochs': 150,
        }
    }
    epochs = task_conf[task]['epochs']
    output_size = task_conf[task]['output_size']
    evaluate_fn = evaluate_codes
    hyper_params['output_dim'] = output_size

    model = Model(config, hyper_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    loss_fn = medical_codes_loss
    if task == 'd':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule_fn)
    else:
        scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                         task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params num: {pytorch_total_params}")
    alpha = torch.tensor(0.98, device=device, requires_grad=True)
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        loss = 0
        steps = len(train_data)
        st = time.time()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            visit_codes, pro_codes, visit_lens, y, adm_time, dis_time = train_data[step]
            output, loss_extra = model(visit_codes, pro_codes, visit_lens, adm_time, dis_time)
            # loss = alpha * loss_fn(output, y) + (1 - alpha) * loss_extra
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, loss), end='')
        scheduler.step()
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, loss))
        evaluate_fn(model, valid_data)

    print("Test Start:")
    evaluate_fn(model, test_data)
