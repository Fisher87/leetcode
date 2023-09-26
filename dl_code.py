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
