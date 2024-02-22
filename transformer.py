#!/usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2024 Fisher. All rights reserved.
#   
#   文件名称：transformer.py
#   创 建 者：YuLianghua
#   创建日期：2024年01月24日
#   描    述：
#
#================================================================

# transformer 模型

import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(d_model))
        self.beta = torch.nn.Parameter(torch.zeros(d_modl))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        out = (x-mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class scaleDotAttention(torch.nn.Module):
    def __init__(self, ):
        super(scaleDotAttention, self).__init__()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        bsz, num_head, seq_len, head_size = q.shape
        score = (q @ k.transpose(2, 3)) / math.sqrt(head_size)   # @ is equal torch.matmul()
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

class MHA(torch.nn.Module):
    def __init__(self, d_model, n_head, device):
        super(MHA, self).__init__()
        self.n_head = n_head
        self.drop_prob = drop_prob
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.attention = scaleDotAttention()

    def forward(self, q, k, v, src_mask):
        # 1. dot product with weight matrics
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        # 2. split tensor by num head
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, src_mask)

        # 4. concat and pass to linear layer
        out = self.out_proj( self.concat(out) )

        return out attention

    def split(self, tensor):
        bsz, seq_len, d_model = tensor.shape
        tensor = tensor.view(bsz, seq_len, self.n_head, -1).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        bsz, n_head, seq_len, head_size = tensor.shape
        d_model = head_size * n_head
        # tensor = tensor.transpose(1, 2).continguous().view(bsz, seq_len, d_model)
        tensor = tensor.transpose(1, 2).reshape(bsz, seq_len, d_model)
        return tensor

class FFN(torch.nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(FFN, self).__init__()
        self.up_proj = torch.nn.Linear(d_model, hidden)
        self.down_proj = torch.nn.Linear(hidden, d_model)
        self.act = torch.nn.Relu()
        self.dropout = torch.nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.dropout( self.relu( self.up_proj(x) ) )
        x = self.down_proj( x )
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, device):
        super(EncoderLayer, self).__init__()
        self.attention = MHA(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = torch.nn.Dropout(p=drop_prob)

        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = torch.nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        residual = x
        x, attention = self.attention(x)
        x = self.norm1(self.dropout1(x) + residual)

        residual = x
        x = self.ffn(x)
        x = self.norm2(self.dropout2(x) + residual)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

class TransformerEmbedding():
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)

class Encoder(torch.nn.Module):
    def __init__(self, d_model, n_head, max_len, vocab_size, drop_prob, n_layers, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            vocab_size = vocab_size,
            drop_prob = drop_prob,
            device = device)
        self.layers = [
            EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob, device=device)
                for _ in range(n_layers)
        ]

    def forward(self, src, src_mask):
        x = self.emb(x)   # [bsz, s, H]
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, device):
        super(DecoderLayer, self).__init__()
        self.self_attn = MHA(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=drop_prob)

        self.enc_dec_attn = MHA(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(p=drop_prob)

        self.ffn = FFN(d_model=d_model, hidden=ffn_hidden)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = torch.nn.Dropout(p=drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # self MHA
        residual = trg
        trg = self.self_attn(q=trg, k=trg, v=trg, mask=trg_mask)
        trg = self.norm1( self.dropout1(trg) + residual )

        # encoder decoder cross attention
        if enc_src is not None:
            residual = trg
            trg = self.enc_dec_attn(q=enc_src, k=trg, v=trg, mask=src_mask)
            trg = self.norm2( self.dropout2(trg) + residual )

        # ffn
        residual = trg
        trg = self.ffn(trg)
        trg = self.norm3( self.dropout3(trg) + residual )

        return trg

class Decoder(torch.nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, vocab_size, drop_prob, n_layers, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        vocab_size=vocab_size,
                                        drop_prob=drop_prob,
                                        device = device)
        self.layers = torch.nn.ModuleList([
            DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob, device=device)
                for _ in range(n_layers)
        ])
        self.linear = torch.nn.Linear(d_model, vocab_size)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output

class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.src_pad_idx = config.src_pad_idx
        self.trg_pad_idx = config.trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.encoder = Encoder(
            d_model = config.d_model,
            n_head = config.n_head,
            max_len= config.max_len,
            ffn_hidden=config.ffn_hidden,
            vocab_size=config.vocab_size,
            drop_prob =config.drop_prob,
            n_layers = config.n_layers,
            device = device)
        self.decoder = Decoder(
            d_model = config.d_model,
            n_head = config.n_head,
            max_len= config.max_len,
            ffn_hidden=config.ffn_hidden,
            vocab_size=config.vocab_size,
            drop_prob =config.drop_prob,
            n_layers = config.n_layers,
            device = device)

    def forward(self, src, trg):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = (trg != self.trg_pad_idx)
        enc_src = self.encoder(src, src_mask)
        ouput  = self.decoder(trg, enc_src, trg_mask, src_mask)
        return ouput

    def src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def trg_mask(self, trg):
        # target 的 mask 比较重要，而且相对复杂些，要同时考虑 长度 padding 及 casual过程
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
