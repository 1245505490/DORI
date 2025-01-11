import os
import pickle
import re
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import nn

stopwords_set = set(stopwords.words('english'))
word_pattern = re.compile(r'[^A-Za-z0-9]')


def extract_word(text):
    """Clean and tokenize text, removing stopwords."""
    text = word_pattern.sub(' ', text.strip().lower())
    return [word for word in word_tokenize(text) if word not in stopwords_set]


def getWordEmb(dataset, icd9_map, code_map, hidden_size):
    dictionary = OrderedDict([("[UNK]", 0)])
    disease_encoded = {}
    for icd9, cid in code_map.items():
        if icd9 == 'PAD': continue
        words = extract_word(icd9_map.get(icd9, ""))
        encoded = [dictionary.setdefault(word, len(dictionary)) for word in words]
        disease_encoded[cid] = encoded

    config = SimpleNamespace(
        vocab_size=len(dictionary) + 1,
        hidden_size=hidden_size,
        num_layers=4,
        num_heads=8,
        intermediate_size=256,
        max_position_embeddings=512
    )
    model = SimpleBERTModel(config)
    model.eval()

    embeddings = OrderedDict()
    for i, (icd9, cid) in enumerate(code_map.items(), 1):
        print(f'\r\t{i} / {len(code_map)}', end='')
        token_ids = torch.tensor([disease_encoded.get(cid, [0])], dtype=torch.long)
        with torch.no_grad():
            sentence_embedding = model(token_ids).squeeze(0).mean(dim=0)
        embeddings[cid] = sentence_embedding.cpu()

    embeddings[0] = torch.zeros_like(embeddings[1], dtype=embeddings[1].dtype)
    pickle.dump(embeddings, open(os.path.join(dataset, 'disease_emb.pkl'), 'wb'))
    print("\ndone!")


class SimpleBERTModel(nn.Module):
    def __init__(self, config):
        super(SimpleBERTModel, self).__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_heads,
                                        dim_feedforward=config.intermediate_size)
             for _ in range(config.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).to(input_ids.device)
        word_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        for layer in self.encoder_layers:
            embeddings = layer(embeddings)

        return embeddings


def parse_icd9_range(range_: str):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        # start:01 end:09
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        # start:840 end:845
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start + 1
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    three_level_code_set = {code.split('.')[0] for code in code_map if code != 'PAD'}
    icd9_range = list(open(os.path.join(path, 'icd9.txt'), 'r', encoding='utf-8').readlines())
    three_level_dict = {}
    level1, level2, level3 = 1, 1, 1
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    level4 = 1
    code_level = {}
    miss_code = []
    for code in code_map:
        three_level_code = code.split('.')[0]
        if three_level_code in three_level_dict:
            three_level = three_level_dict[three_level_code]
            code_level[code] = three_level + [level4]
            level4 += 1
        else:
            miss_code.append(three_level_code)
            code_level[code] = [0, 0, 0, 0]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]
    print(miss_code)
    return code_level_matrix


def generate_code_code_adjacent(pids, code_map, patient_admission, admission_codes_encoded, p_admission_codes_encoded,
                                code_num,
                                pro_code_num,
                                threshold, all_patient, all_admission_codes):
    print('generating code code adjacent matrix ...')
    n = code_num
    m = pro_code_num
    adj = np.zeros((n + 1, n + 1), dtype=float)
    total_adj = np.zeros((m + 1, m + 1), dtype=float)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        admissions = patient_admission[pid]
        for k, admission in enumerate(admissions[:-1]):
            codes = admission_codes_encoded[admission['adm_id']]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1
                    adj[c_j, c_i] += 1
                    total_adj[c_i, c_j] += 1
                    total_adj[c_j, c_i] += 1
            if p_admission_codes_encoded is not None:
                codes = p_admission_codes_encoded[admission['adm_id']]
                if codes[0] != 0:
                    for row in range(len(codes) - 1):
                        for col in range(row + 1, len(codes)):
                            c_i = codes[row]
                            c_j = codes[col]
                            total_adj[c_i, c_j] += 1
                            total_adj[c_j, c_i] += 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    print("Remaining pids:")
    if all_patient is not None:
        all_pids = all_patient.keys()
        remain_pids = [item for item in all_pids if item not in pids]
        for i, pid in enumerate(remain_pids):
            print('\r\t%d / %d' % (i, len(remain_pids)), end='')
            admissions = all_patient[pid]
            for k, admission in enumerate(admissions):
                adm_id = admission['adm_id']
                if adm_id not in all_admission_codes:
                    continue
                icd9_list = all_admission_codes[adm_id]
                codes = [code_map[icd9] for icd9 in icd9_list if icd9 in code_map]
                for row in range(len(codes) - 1):
                    for col in range(row + 1, len(codes)):
                        c_i = codes[row]
                        c_j = codes[col]
                        adj[c_i, c_j] += 1
                        adj[c_j, c_i] += 1
                        total_adj[c_i, c_j] += 1
                        total_adj[c_j, c_i] += 1
        print('\r\t%d / %d' % (len(remain_pids), len(remain_pids)))
    return screening(adj, threshold), screening(total_adj, threshold)


def normalize_adj(adj):
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s
    return result


def screening(adj: np.ndarray, threshold):
    norm_adj = normalize_adj(adj)
    a = norm_adj < threshold
    b = adj.sum(axis=-1, keepdims=True) > (1 / threshold)
    adj[np.logical_and(a, b)] = 0
    return adj
