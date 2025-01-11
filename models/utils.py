import torch


def seq_mask(visit_lens, max_seq_len):
    n_sample = len(visit_lens)
    visit_mask = torch.zeros((n_sample, max_seq_len), dtype=torch.int32, device=visit_lens.device)
    for i, lens in enumerate(visit_lens):
        visit_mask[i, :lens] = 1
    return visit_mask


def get_multmask(visit_mask):
    mult = (1 - visit_mask).to(dtype=torch.bool, device=visit_mask.device)
    return mult


def final_mask(visit_lens, max_seq_len):
    n_sample = len(visit_lens)
    visit_mask = torch.zeros((n_sample, max_seq_len), dtype=torch.int32, device=visit_lens.device)
    for i in range(n_sample):
        visit_mask[i, visit_lens[i] - 1] = 1
    return visit_mask


def masked_softmax(inputs, mask):

    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]

    exp = torch.exp(inputs) * mask

    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    sum_exp[sum_exp == 0] = 1
    result = torch.div(exp, sum_exp)
    return result


def normalize_adjacency_matrix(adj):
    row_sum = torch.sum(adj, dim=-1, keepdim=True)
    row_sum[row_sum == 0] = 1
    return adj / row_sum

def my_pad_to_length(seq, max_len):
    length = seq.size(1)
    dim = seq.size(-1)
    if length == max_len:
        return seq
    else:
        pad_len = max_len - length
        padding = seq.new_zeros((1, pad_len, dim))
        return torch.cat([seq, padding], dim=1)