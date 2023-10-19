#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 Fisher. All rights reserved.
#   
#   文件名称：generate.py
#   创 建 者：YuLianghua
#   创建日期：2023年10月19日
#   描    述：
#
#================================================================

class Llama:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts, max_gen_len, temperature, top_p):
        bsz = len(prompts)
        
        prompt_tokens = [self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(self.model.max_seq_len, max_prompt_size+max_gen_len)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperatrue, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token 
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []

        for i, t in enumerate(tokens.tolist()):
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded

def sample_top_p(probs, p): # probs: [0.1, 0.3, 0.1, 0.05, 0.2, 0.25], p=0.6
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)   # probs_sort: [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
    probs_sum = torch.cumsum(probs_sort, dim=-1)                         # probs_sum:  [0.3, 0.55, 0.75, 0.85, 0.95, 1.0]
    mask = probs_sum - probs_sort > p                                    #      mask:  [False, False, False, True, True, True]    
    probs_sort[mask] = 0.0                                               # probs_sort: [0.3, 0.25, 0.2, 0.0, 0.0, 0.0]
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
