#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2024 Fisher. All rights reserved.
#   
#   文件名称：moe.py
#   创 建 者：YuLianghua
#   创建日期：2024年01月22日
#   描    述：
#
#================================================================

class MoE(torch.Module):
    def __init__(self, config, num_experts=8):
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape  # [b, s, h]
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=troch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # [b*s, topk]

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weigths.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size*sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  # [n_experts, topk, b*s]

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0]==0:
                continue

            top_x_list = top_x.tolist()
            idx_list = idx.to_list()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, seqid, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, seqid, nth_expert] * expert(
                inputs[batch_idx]
            )
        return results
