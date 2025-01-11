from models.ODE import ODENet
from models.transformer import *
from models.utils import masked_softmax, my_pad_to_length


class HEmbeddingLayer(nn.Module):
    def __init__(self, code_dims, code_levels, leaf_list, ancestor_list, attention_dim_size, code_num_in_levels):
        super().__init__()
        self.code_num = len(code_levels)
        self.code_levels = code_levels
        self.code_num_in_levels = code_num_in_levels
        self.level_num = len(self.code_num_in_levels)
        self.level_embeddings = nn.ModuleList([nn.Embedding(code_num + 1, code_dim, padding_idx=0)
                                               for code_num, code_dim in
                                               zip(self.code_num_in_levels, code_dims)])
        code_dim = code_dims[-1]
        self.leaf_list = leaf_list
        self.ancestor_list = ancestor_list
        self.linear1 = nn.Linear(2 * code_dim, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self):
        leaves_emb = self.level_embeddings[-1](self.leaf_list)
        h_emb_list = []
        for level in range(self.level_num):
            h_emb = self.level_embeddings[level](self.ancestor_list[:, level])
            h_emb_list.append(h_emb)
        ancestor_emb = torch.stack(h_emb_list, dim=1)
        x = torch.cat([leaves_emb, ancestor_emb], dim=-1)
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = torch.softmax(x, dim=1)
        x = (x * ancestor_emb).sum(dim=1)
        return x


class LEmbeddingLayer(nn.Module):
    def __init__(self, beta, leaf, leaf_mask, code_dim, attention_dim_size):
        super().__init__()
        self.beta = beta.unsqueeze(2)
        self.leaf = leaf
        self.leaf_mask = leaf_mask.unsqueeze(2)
        self.linear1 = nn.Linear(2 * code_dim, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)
        self.leaky = nn.LeakyReLU()

    def forward(self, embedding):
        leaf_embedding = embedding.unsqueeze(1)
        leaf_embedding = leaf_embedding.repeat(1, self.leaf.size(-1), 1)
        neighbors = F.embedding(self.leaf, embedding, padding_idx=0)
        e = torch.cat([leaf_embedding * self.leaf_mask, neighbors * self.leaf_mask], dim=-1)
        e = self.linear1(e)
        e = self.leaky(e)
        e = self.linear2(e)
        mask_attn = (1.0 - self.leaf_mask) * (-1e30)
        e += mask_attn
        x = torch.softmax(e, dim=1)
        final_emb = (x * neighbors * self.leaf_mask * self.beta).sum(dim=1)
        return final_emb


class DiseaseWordEmbedding(nn.Module):
    def __init__(self, disease_emb):
        super(DiseaseWordEmbedding, self).__init__()
        self.disease_emb = disease_emb

    def forward(self, final_emb):
        final_emb = torch.cat([final_emb, self.disease_emb], dim=-1)
        return final_emb


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.dense_dag = nn.Linear(hidden_size, intermediate_size)
        self.linear = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states, hidden_states_dag):
        hidden_states_ = self.dense(hidden_states)
        hidden_states_dag_ = self.dense_dag(hidden_states_dag)
        hidden_states = self.linear(F.relu(hidden_states_ + hidden_states_dag_))
        return hidden_states


class VisitEmbedding(nn.Module):
    def __init__(self, max_visit_len):
        super().__init__()
        self.max_seq_len = max_visit_len

    def forward(self, visit_codes_embedding, visit_codes, visit_mask):
        visit_codes_mask = torch.unsqueeze(visit_codes > 0, dim=-1).to(dtype=visit_codes_embedding.dtype,
                                                                       device=visit_codes.device)
        visit_codes_embedding *= visit_codes_mask
        visit_codes_num = torch.unsqueeze(
            torch.sum((visit_codes > 0).to(dtype=visit_codes_embedding.dtype), dim=-1), dim=-1)
        sum_visit_codes_embedding = torch.sum(visit_codes_embedding, dim=-2)
        visit_codes_num[visit_codes_num == 0] = 1
        visits_embeddings = sum_visit_codes_embedding / visit_codes_num
        visits_embeddings = visits_embeddings * visit_mask.unsqueeze(-1)
        return visits_embeddings


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim=model_dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


def padding_mask(seq_q, seq_k):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class Encoder(nn.Module):
    def __init__(self, max_visit_len, num_layers, model_dim, num_heads, ffn_dim, time_dim,
                 dropout, device):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
            range(num_layers)
        ])
        self.time_layer = TimeFeedforwardEmbedding(model_dim)
        self.pos_embedding = PositionalEncoding(model_dim=model_dim, max_visit_len=max_visit_len)
        self.tanh = nn.Tanh()

        self.odeNet_los = ODENet(device, model_dim, model_dim)
        self.odeNet_interval = ODENet(device, model_dim, model_dim,
                                      output_dim=model_dim, augment_dim=10, time_dependent=True)
        self.LayerNorm = BertLayerNorm(model_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def is_strictly_increasing(self, time_i, lens):
        for i in range(1, lens):
            if time_i[i] <= time_i[i - 1]:
                return False
        return True

    def getOde(self, time, odeNet: ODENet, visits_embeddings, visit_lens):
        batch_size = visits_embeddings.size(0)
        max_visit_len = visits_embeddings.size(1)
        embeddings_list = []
        y0_batch = visits_embeddings[:, 0, :]
        for i in range(batch_size):
            y0_i = y0_batch[i]  # [model_dim]
            time_i = time[i][:visit_lens[i]]  # [visit_len]
            if not self.is_strictly_increasing(time_i, visit_lens[i]):
                time_i = torch.linspace(0, 1, visit_lens[i])
            y0_i = y0_i.unsqueeze(0)  # [1, model_dim]
            interval_sol_i = odeNet(y0_i, time_i).permute(1, 0, 2)  # [1,visitlen,model_dim]
            interval_sol_i_padded = my_pad_to_length(interval_sol_i, max_visit_len)
            embeddings_list.append(interval_sol_i_padded)
        interval_embeddings = torch.stack(embeddings_list, dim=0).squeeze()
        return interval_embeddings

    def forward(self, visits_embeddings, visit_mask, visit_lens, adm_time, dis_time):
        v_mask = visit_mask.unsqueeze(-1)
        output_pos, ind_pos = self.pos_embedding(visit_lens.unsqueeze(-1))
        interval_embeddings = self.getOde(adm_time, self.odeNet_interval, visits_embeddings, visit_lens) + self.getOde(
            dis_time, self.odeNet_interval,
            visits_embeddings,
            visit_lens)
        los_embeddings = self.odeNet_los(visits_embeddings)
        time_embedding = self.time_layer(adm_time, visit_mask)
        output = visits_embeddings + output_pos + interval_embeddings + los_embeddings + time_embedding
        output = output * v_mask
        att_mask = padding_mask(ind_pos, ind_pos)
        for encoder in self.encoder_layers:
            output, _ = encoder(output, att_mask)
            output = self.dropout(output)

        return output


class TimeFeedforwardEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeFeedforwardEmbedding, self).__init__()
        self.fc1 = nn.Linear(1, embedding_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim // 2, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, time_tensor, mask=None):
        batch_size, max_visit_num = time_tensor.size()
        time_flat = time_tensor.view(-1, 1)
        embedding_flat = self.fc1(time_flat)
        embedding_flat = self.relu(embedding_flat)
        embedding_flat = self.fc2(embedding_flat)
        embedding = embedding_flat.view(batch_size, max_visit_num, self.embedding_dim)

        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [batch_size, max_visit_num, 1]
            embedding = embedding * mask  # Zero out embeddings where mask == 0

        return embedding  # [batch_size, max_visit_num, embedding_dim]


class Attention(nn.Module):
    def __init__(self, input_size, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.u_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(attention_dim, 1))))
        self.w_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(input_size, attention_dim))))

    def forward(self, x, mask=None):
        t = torch.matmul(x, self.w_omega)
        vu = torch.squeeze(torch.tensordot(t, self.u_omega, dims=1), dim=-1)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu)
        alphas = alphas.unsqueeze(-1)
        # alphas = alphas / (torch.sum(alphas, 1, keepdim=True) + 1e-10)
        output = torch.sum(x * alphas, dim=-2)  # (batch_size, code_dim )
        return output

class MyGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dims=256):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(input_dim, output_dim))))
        self.fai = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.fai2 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.linear3 = nn.Linear(input_dim, output_dim)
        self.intermediate = BertIntermediate(output_dim, intermediate_dims)

    def propagate3(self, X):
        N, _ = X.size()
        I = torch.eye(N).to(X.device)
        Z = torch.matmul(I, X)
        Z = self.linear3(Z)
        return F.relu(Z)

    def propagate1(self, adj, X):
        N, N = adj.size()
        I = torch.eye(N).to(adj.device)
        A_hat = self.fai * adj + (1 - self.fai) * I
        A_norm = F.normalize(A_hat, p=1, dim=1)
        Z = torch.matmul(A_norm, X)
        Z = self.linear(Z)
        return F.relu(Z)

    def propagate2(self, adj, X):
        N, N = adj.size()
        I = torch.eye(N).to(adj.device)
        A_hat = self.fai2 * adj + (1 - self.fai2) * I
        A_norm = F.normalize(A_hat, p=1, dim=1)
        Z = torch.matmul(A_norm, X)
        Z = self.linear2(Z)
        return F.relu(Z)

    #

    def forward(self, adj1, adj2, X):
        H1 = self.propagate1(adj1, X)
        H2 = self.propagate2(adj2, X)
        H3 = self.propagate3(X)
        return F.layer_norm(self.intermediate(H1, H2) + H3, normalized_shape=(H1.size(-1),))


class MyRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MyGCNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(MyGCNLayer(hidden_dim, hidden_dim))
        self.layers.append(MyGCNLayer(hidden_dim, output_dim))

    def forward(self, A1, A2, X):
        for layer in self.layers:
            X = layer(A1, A2, X)
        return X


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.FloatTensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.num_inds = num_inds

    def forward(self, input):
        X, attn_mask = input
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, attn_mask)  # [B*V, num_inds, dim]
        attn_mask = attn_mask.transpose(-2, -1)
        return self.mab1(X, H, attn_mask)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, attn_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        ############################################################
        attn_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if attn_mask is not None:
            attn_mask = attn_mask.view_as(attn_score)
            if attn_mask.dtype == torch.bool:
                attn_score.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_score += attn_mask
        ############################################################

        A = torch.softmax(attn_score, 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
