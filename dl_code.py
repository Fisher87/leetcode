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

from einops import rearrange

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

class LayerNorm(nn.Module):
    def __init__():
        pass

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

############################################################################################################
# multiquery attention

class MQA(nn.Module):
    def __init__(self, d_model, head_nums, device=None):
        super().__init__()
        self.d_model = d_model
        self.head_nums = head_nums
        self.head_dim = d_model//head_nums

        self.Wqkv = nn.Linear(
            d_model,
            d_model + 2 * self.head_dim
            device=device)
        self.out_proj = nn.Linear(d_model, d_model, device=device)

    def forward(self, x, 
                attn_bias=None,
                attention_mask=None,
                is_causal=True,
                needs_weights=False):
        # qkv
        qkv = self.Wqkv(x)
        query, key, value = qkv.split( [self.d_model, self.head_dim, self.head_dim], dim=2)
        key_padding_mask = attention_mask

        context, attn_weights, past_key_value = self.attn_fn(query, key, value, 
                                                     self.head_nums, self.softmax_scale, multiquery=True)
        return self.out_proj(context), attn_weights

    def attn_fn(self, query, key, value, n_heads, softmax_scale=None, multiquery=False):
        q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
        kv_n_heads = 1 if multiquery else n_heads
        k = rearrange(key, 'b s (h d) -> b h d s', h=kv_n_heads)
        v = rearrange(value, 'b s (h d) -> b h s d', h = kv_n_heads)

        attn_weights = q.matmul(k) * softmax_scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = attn_weights.matmul(v)
        out = rearrange(out, 'b h s d -> b s (h d)')
        return out, attn_weight

############################################################################################################
# group query attention

class GQA(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, num_key_value_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_group_size = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads*self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=False)

    def repeat_kv(self, hidden_states, n_rep):
        return torch.repeat_interleave(hidden_states, n_rep, 1)

    def forward(self, hidden_states, attention_mask):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states= value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = self.repeat_kv(key_states, self.num_key_value_group_size)
        value_states=self.repeat_kv(value_states, self.num_key_value_group_size)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

############################################################################################################

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)    # 构建causal mask
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

############################################################################################################

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        # -torch.arange(0,dim,2)/dim * math.log(10000.)
        div_term = torch.exp( torch.arange(0, dim, 2)*( -math.log(10000.)/dim ) )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        l, *_ = x.shape
        return self.pe[:l, :].unsqueeze(1)

############################################################################################################

class GLobalPointerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        '''
        y_true / y_pred : [bsz, num_heads, max_seq_len, max_seq_len]
        '''
        bsz, num_heads, max_seq_len = y_pred.shape[:3] 
        y_true = y_true.reshape([bsz*num_heads, -1])
        y_pred = y_pred.reshape([bsz*num_heads, -1])

        y_pred = (1 - 2*y_true) * y_pred     # 交换，将负样本得分s_i 保持不变，正样本得分s_j 变为 -s_j
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1-y_true) * 1e12

        zero_vec = torch.zeros([bsz*num_heads, 1], device=y_pred.device)
        y_pred_neg = torch.cat([y_pred_neg, zero_vec], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zero_vec], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return torch.mean(neg_loss + pos_loss)

############################################################################################################

# 在不关闭进程的情况下回收显存
from accelerate.utils import release_memory
def release(model)
    model = model.to('cpu')
    model = release_memory(model)
