#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 Fisher. All rights reserved.
#   
#   文件名称：dl_code.py
#   创 建 者：YuLianghua
#   创建日期：2023年09月25日
#   描    述：
#
#================================================================

def viterbi(obs, states, start_prob, trans_prob, emission_prob):
    T = len(obs)
    N = len(states)

    viterbi_mat = np.zeros((N, T))
    path_mat = np.zeros((N, T), dtype=int)

    viterbi_mat[:, 0] = start_prob * emission_prob[:, obs[0]]
    path_mat[:, 0] = 0
    for t in range(1, T):
        for s in range(N):
            prob = viterbi_mat[:, t-1] + trans[:, s] + emission_prob[s, obs[t]]
            viterbi_mat[s, t] = np.max(prob)
            path_mat[s, t] = np.argmax(prob)

    best_path = [np.argmax(viterbi_mat[:, -1])]
    for t in range(T-1, 0, -1):
        best_path.insert(0, path_mat[best_path[0], t])

    return best_path

############################################################################################################

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, scale=1, base=10000, device=None):
        super().__init__()
        self.scale = scale
        # 构建基值向量
        inv_freq = 1.0 / ( base**(torch.arange(0, dim//2, 2).float().to(device)/dim) )
        self.register_buffer('inv_freq', inv_freq)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t /= scale

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.range(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            t /= scale
            freqs = torch.enisum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
            return (
                self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
                self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
            )
        
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=x1.ndim-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_id):
    q, k = (q*cos) + (rotate_half(q)*sin), (k*cos)+(rotate_half(k)*sin)

############################################################################################################

# RMSNorm: x = \frac{x}{\sum sqrt(x_{i}^2)}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

############################################################################################################

class BiLstmCRF(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, tag2id):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)

        self.word_embeds = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, num_layers=1, bidirection=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag2id[START_TAG], :] = -10000
        self.transitions.data[:, tag2id[STOP_TAG]] = -10000

        self.hidden = (    # 初始的hidden
            torch.randn(2, 1, self.hidden_size//2), torch.randn(2, 1, self.hidden_size//2))

############################################################################################################

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(f_size, self.embed_dim))
                    for f_size in filter_sizes
            ]
        )
        self.fc = nn.Linear(n_filters*len(filter_sizes), output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.fc(cat)
        return logits

############################################################################################################

class SelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, n_heads):
        super().__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = n_heads

        self.q_linear = nn.Linear(dim_in, dim_k)
        self.k_linear = nn.Linear(dim_in, dim_k)
        self.v_linear = nn.Linear(dim_in, dim_v)
        self.o_linear = nn.Linear(dim_in, dim_in)

    def forward(self, x):
        bs, seq_len, dim_in = x.shape()
        head_dim = self.dim_k // self.num_heads
        q = self.q_linear(x).reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_linear(x).reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_linear(x).reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * 1/sqrt(head_dim)
        dist = torch.softmax(dist, dim=-1)
        attn = torch.matmul(dist, v)
        attn = attn.transpose(1, 2).reshape(bs, seq_len, -1)

        return attn
