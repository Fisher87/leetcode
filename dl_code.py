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

import torch
import torch.nn.functional as F

# LaryerNorm x = \gamma * \frac{x-mean}{sqrt(std)} + \beta
class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 使用示例
input_tensor = torch.randn(3, 4, 5)  # 输入 tensor 大小为 (batch_size, seq_len, features)
layer_norm = LayerNorm(5)  # LayerNorm 将 features 设置为 5
output = layer_norm(input_tensor)
print(output)

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

############################################################################################################

# 在不加载模型参数权重的情况下查看模型结构信息
from accelerate.utils import get_balanced_memory, infer_auto_device_map, find_tied_parameters
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate import dispatch_model
from transformers import AutoTokenizer, AutoModel, AutoConfig

## 正常加载方式，会消耗显存
# model = AutoModelForCausalLM.from_pretrained(
#     pretrained_model_name_or_path=model_path,
#     load_in_8bit=True,
#     torch_dtype=torch.float16,
#     device_map='auto',
#     max_memory=None,
#     trust_remote_code=True)

# 不消耗显存
mconfig = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(mconfig, trust_remote_code=True)

############################################################################################################

# ROPE 位置编码
def precompute_freqs_cis(dim, max_length, base=10000):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim//2)].float() /dim ))
    t = torch.arange(max_length, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 或者

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super(LlamaRotaryEmbedding, self).__init__() 

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings 
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )

def rotate_half(x, interleaved=False):
    if not interleaved: # 不交错
        # x1, x2 = x.chunk(2, dim=-1)
        dim = x.shape[-1]
        x1, x2 = x[..., :dim//2], x[..., dim//2:]
        return torch.cat([-x2, x1], dim=-1)
    else: # 交错
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), '... d two -> ... (d two)', two=2)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)     # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)   # [bsz, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return (q_embed, k_embed)

# query_states -> [bsz, num_head, seq_len, head_dim]
# position_ids -> [bsz, seq_len]
rotary_embe = LlamaRotaryEmbedding(1024)
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

############################################################################################################
import torch
import torch.nn as nn

class SlidingWindowAttention(nn.Module):
    def __init__(self, input_size, window_size, hidden_size):
        super(SlidingWindowAttention, self).__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.hidden_size = hidden_size

        # Linear layer to project input embeddings to the desired hidden size
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Linear layers for computing attention scores
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        output = torch.zeros_like(inputs)

        for i in range(0, seq_len, self.window_size):
            window = inputs[:, i:i+self.window_size, :]  # Extract window
            window_len = window.size(1)

            # Project window to hidden size
            window_proj = self.input_projection(window)

            # Compute attention scores
            Q = self.query(window_proj)
            K = self.key(window_proj)
            V = self.value(window_proj)

            # Calculate attention weights
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
            attn_weights = self.softmax(scores)

            # Apply attention to values
            window_output = torch.matmul(attn_weights, V)

            # Concatenate window outputs
            output[:, i:i+window_len, :] = window_output

        # Project concatenated output to original input size
        output = self.output_projection(output)

        return output

# Example usage:
input_size = 512
window_size = 5
hidden_size = 256
seq_len = 20
batch_size = 32

inputs = torch.randn(batch_size, seq_len, input_size)
sliding_attention = SlidingWindowAttention(input_size, window_size, hidden_size)
output = sliding_attention(inputs)

############################################################################################################
# beam_search decode
class BeamSearchNode:
    def __init__(self, sequence, score):
        self.sequence = sequence
        self.score = score

def next_words_probs_infer(sequence):
    probs = model.forward(sequence)

def beam_search(initial_sequence, next_words_probs_infer_func, beam_size, max_sequence_length):
    # 初始化初始节点，且分数为1
    initial_node = BeamSearchNode(sequence=initial_sequence, score=1.0)
    candidates = [initial_node]
    final_candidates = []  # 最终的候选序列
    # 只要候选节点列表不为空，且 final_candidates 中的候选节点数量还没有达到指定的束宽度，就继续进行搜索
    while candidates and len(final_candidates) < num_beams:
        # 候选节点排序
        candidates.sort(key=lambda x: -x.score)
        current_node = candidates.pop(0)
        # 当节点序列末尾生成结束符号（如"<end>"），或者当生成的序列长度达到最大限制时终止节点的扩展
        if current_node.sequence[-1] == "<end>" or len(current_node.sequence) >= max_sequence_length:
            final_candidates.append(current_node)
        else:
            # 获取下一个token的概率，我们的例子返回的是固定的概率
            next_words_probs = next_word_probs_func(current_node.sequence)
            # 生成新的候选序列，并计算分数
            for next_word, next_word_prob in next_words_probs.items():
                new_sequence = current_node.sequence + [next_word]
                new_score = current_node.score * next_word_prob
                new_node = BeamSearchNode(sequence=new_sequence, score=new_score)
                candidates.append(new_node)

    return [candidate.sequence for candidate in final_candidates]

############################################################################################################
# cross entropy
def cross_entropy(targets, predictions, eps=1e-6):
    predictions = np.clip(predictions, eps, 1.0-eps)
    ce = -np.sum(targets * np.log(predictions + eps))
    return ce

predictions = np.array([[0.6, 0.2, 0.2], [0.3, 0.4, 0.3]])
targets = np.array([[1, 0, 0], [0, 1, 0]])
loss = cross_entropy(predictions, targets)

# softmax cross entropy
def softmax_cross_entropy(targets, predictions, eps=1e-6):
    exp_x = np.exp(predictions-np.max(predictions, axis=-1, keepdim=True))
    pred_probs = exp_x / np.sum(exp_x, axis=-1, keepdim=True)
    pred_probs = np.clip(pred_probs, eps, 1.0-eps)
    ce = -np.sum(targets * np.log(pred_probs + eps))
    return ce
predictions = np.array([[2.0, 1.0, 0.1], [1.0, 0.9, 0.8]])
targets = np.array([[1, 0, 0], [0, 1, 0]])
loss = softmax_cross_entropy(predictions, targets)
