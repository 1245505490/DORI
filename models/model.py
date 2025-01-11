from types import SimpleNamespace

from models.layers import *
from models.utils import seq_mask, normalize_adjacency_matrix


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        return output


class Model(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        device = config['device']
        self.use_proc = config['use_proc']
        self.fe = FeatureExtractor(config, hyper_params).to(device)
        if self.use_proc:
            self.classifier = Classifier(input_size=hyper_params['hidden_dim'] * 2,
                                         output_size=hyper_params['output_dim'],
                                         dropout_rate=hyper_params['dropout']).to(device)
        else:
            self.classifier = Classifier(input_size=hyper_params['hidden_dim'], output_size=hyper_params['output_dim'],
                                         dropout_rate=hyper_params['dropout']).to(device)

    def getCodeEmbedding(self):
        return self.fe.code_embeddings

    def forward(self, visit_codes, pro_codes, visit_lens, adm_time, dis_time):
        inputs = {
            'visit_codes': visit_codes,
            'visit_lens': visit_lens,
            'pro_codes': pro_codes,
            'adm_time': adm_time,
            'dis_time': dis_time
        }
        inputs = SimpleNamespace(**inputs)
        if self.training:
            output, loss_extra = self.fe(inputs)
            output = self.classifier(output)
            return output, loss_extra
        else:
            output = self.fe(inputs)
            output = self.classifier(output)
            return output


class FeatureExtractor(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        self.config = config
        self.hyper_params = hyper_params
        self.device = config['device']
        code_dim = hyper_params['code_dims'][-1]
        self.use_proc = config['use_proc']
        self.nhead = hyper_params['num_heads']
        self.num_inds = 16
        self.h_embedding = HEmbeddingLayer(code_dims=hyper_params['code_dims'], code_levels=config['code_levels'],
                                           leaf_list=config['leaf_list'], ancestor_list=config['ancestor_list'],
                                           attention_dim_size=hyper_params['attention_dim'],
                                           code_num_in_levels=config['code_num_in_levels']).to(self.device)
        self.l_embedding = LEmbeddingLayer(beta=config['beta'], leaf=config['leaf'], leaf_mask=config['leaf_mask'],
                                           code_dim=code_dim,
                                           attention_dim_size=hyper_params['attention_dim']).to(self.device)
        self.word_embedding = DiseaseWordEmbedding(config['disease_emb']).to(self.device)
        self.max_visit_len = config['max_visit_seq_len']
        self.cos_matrix = None
        self.code_adj = normalize_adjacency_matrix(config['code_adj']).to(dtype=torch.float32)

        self.visit_embedding_layer = VisitEmbedding(max_visit_len=self.max_visit_len).to(self.device)
        if self.use_proc:
            self.feature_encoder = Encoder(max_visit_len=config['max_visit_seq_len'],
                                           num_layers=hyper_params['num_layers'],
                                           model_dim=hyper_params['hidden_dim'] * 2,
                                           num_heads=hyper_params['num_heads'], ffn_dim=hyper_params['ffn_dim'],
                                           time_dim=hyper_params['time_dim'],
                                           dropout=hyper_params['dropout'], device=self.device).to(self.device)
            self.attention = Attention(2 * hyper_params['hidden_dim'], attention_dim=hyper_params['attention_dim']).to(
                self.device)
        else:
            self.feature_encoder = Encoder(max_visit_len=config['max_visit_seq_len'],
                                           num_layers=hyper_params['num_layers'], model_dim=hyper_params['hidden_dim'],
                                           num_heads=hyper_params['num_heads'], ffn_dim=hyper_params['ffn_dim'],
                                           time_dim=hyper_params['time_dim'],
                                           dropout=hyper_params['dropout'], device=self.device).to(self.device)
            self.attention = Attention(hyper_params['hidden_dim'], attention_dim=hyper_params['attention_dim']).to(
                self.device)
        self.myrgcn = MyRGCN(code_dim + hyper_params['word_dim'], hyper_params['gcn_hidden'], hyper_params['gcn_out'],
                             num_layers=2)
        self.isab = ISAB(dim_in=hyper_params['gcn_out'], dim_out=hyper_params['hidden_dim'],
                         num_heads=hyper_params['num_heads'],
                         num_inds=self.num_inds,
                         ln=True)
        self.a = torch.tensor(0.01, requires_grad=True)

    @staticmethod
    def getCosMatrix(adj):
        norm = adj.norm(dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        normed_adj = adj / norm
        cos_matrix = torch.matmul(normed_adj, normed_adj.T).to(adj.device)

        return cos_matrix

    def set_encoder(self, input, attn_mask, rep=2):
        attn_mask = attn_mask.unsqueeze(2).repeat(1, 1, self.num_inds, 1)  # [b*v,nhead,num_inds,c]
        for i in range(rep):
            input = self.isab([input, attn_mask])
        return input

    def calrefact_Loss(self, H, A):
        H = F.normalize(H, p=2, dim=1)
        Ht = torch.t(H)
        R = torch.matmul(H, Ht)
        loss = torch.norm(R - A, p='fro') ** 2
        return loss

    def forward(self, inputs):
        visit_codes = inputs.visit_codes
        visit_lens = inputs.visit_lens
        pro_codes = inputs.pro_codes
        adm_time = inputs.adm_time
        dis_time = inputs.dis_time
        B, V, C = visit_codes.size()
        visit_mask = seq_mask(visit_lens, self.max_visit_len)
        x_mask = (visit_codes > 0).to(dtype=torch.float, device=visit_codes.device)
        h_emb = self.h_embedding()
        code_embeddings = self.l_embedding(h_emb)
        code_embeddings = self.word_embedding(code_embeddings)
        self.cos_matrix = self.getCosMatrix(code_embeddings)
        Z = self.myrgcn(self.code_adj, self.cos_matrix, code_embeddings)
        loss = self.a * self.calrefact_Loss(self.getCosMatrix(Z), self.cos_matrix)
        code_embeddings = Z
        diag_input = (F.embedding(visit_codes, code_embeddings) * x_mask.unsqueeze(-1)).view(B * V, C, -1)
        diag_mask = x_mask.view(B * V, C).unsqueeze(1).repeat(1, self.nhead, 1)
        diag_encode = self.set_encoder(diag_input, diag_mask).view(B, V, C, -1)
        diag_visits_embeddings = self.visit_embedding_layer(diag_encode, visit_codes, visit_mask)
        visits_embeddings = diag_visits_embeddings
        if self.use_proc:
            proc_mask = (pro_codes > 0).to(dtype=torch.float, device=visit_codes.device)
            proc_input = (F.embedding(pro_codes, code_embeddings) * proc_mask.unsqueeze(-1)).view(B * V, C, -1)
            proc_mask = proc_mask.view(B * V, C).unsqueeze(1).repeat(1, self.nhead, 1)
            proc_encode = self.set_encoder(proc_input, proc_mask).view(B, V, C, -1)
            proc_visits_embeddings = self.visit_embedding_layer(proc_encode, pro_codes, visit_mask)
            visits_embeddings = torch.cat([visits_embeddings, proc_visits_embeddings],
                                          dim=-1)

        features = self.feature_encoder(visits_embeddings, visit_mask, visit_lens, adm_time, dis_time)
        outputs = self.attention(features, visit_mask)
        if self.training:
            return outputs, loss
        else:
            return outputs
