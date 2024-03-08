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

def beam_search(k, decoder):
    k_prev_words = torch.full((k, 1), SOS_TOKEN, dtype=torch.long) # (k, 1)
    # 此时输出序列中只有sos token
    seqs = k_prev_words #(k, 1)
    # 初始化scores向量为0
    top_k_scores = torch.zeros(k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 1
    hidden = torch.zeros(1, k, hidden_size) # h_0: (1, k, hidden_size)
    while True:
        outputs, hidden = decoder(k_prev_words, hidden) # outputs: (k, seq_len, vocab_size)
        next_token_logits = outputs[:,-1,:] # (k, vocab_size)
        if step == 1:
            # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
            top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
        else:
            # 此时要先展开再计算topk，如上图所示。
            # top_k_scores: (k) top_k_words: (k)
            top_k_scores, top_k_words = next_token_logits.view(-1).topk(k, 0, True, True)
        prev_word_inds = top_k_words / vocab_size  # (k)  实际是beam_id, 因为通过view(-1)进行展开
        next_word_inds = top_k_words % vocab_size  # (k)  实际是token_id
        # seqs: (k, step) ==> (k, step+1)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != vocab['<eos>']]
        # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
          complete_seqs.extend(seqs[complete_inds].tolist()) # 加入句子
          complete_seqs_scores.extend(top_k_scores[complete_inds]) # 加入句子对应的累加log_prob
        # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
        k -= len(complete_inds)

        if k == 0: # 完成
            break

        # 更新下一次迭代数据, 仅专注于那些还没完成的句子
        seqs = seqs[incomplete_inds]
        hidden = hidden[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)   #(s, 1) s < k
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1) #(s, 1) s < k

        if step > max_length: # decode太长后，直接break掉
            break
        step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores)) # 寻找score最大的序列
    # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
    seq = complete_seqs[i]

    return seq
