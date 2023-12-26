#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 Fisher. All rights reserved.
#   
#   文件名称：GQA.py
#   创 建 者：YuLianghua
#   创建日期：2023年12月11日
#   描    述：
#
#================================================================


def repeat_kv(hidden_states, n_rep):
    bsz, num_key_value_heads, q_len, head_dim = hidden_states.shape
    if n_rep==1:
        return hidden_state
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads*n_rep, q_len, head_dim)


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

