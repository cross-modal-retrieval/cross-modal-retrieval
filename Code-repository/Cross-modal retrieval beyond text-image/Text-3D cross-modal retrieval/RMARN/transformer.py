import torch
import torch.nn as nn
import math
import pickle as pkl
from torchsummary import summary


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # self.register_buffer('bias', torch.tril(
        #     torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)     # b, s, h
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (y.transpose(1, 2).contiguous().view(B, T, C))
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Config:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, device):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.device = device

    def __str__(self):
        return f"voc: {self.vocab_size}, max_len: {self.block_size}, " \
               f"layer: {self.n_layer}, head: {self.n_head}, emb: {self.n_embd}"

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'device': self.device
        }


if __name__ == '__main__':
    conf = Config(
        vocab_size=5500,
        block_size=512,
        n_layer=3,
        n_head=2,
        n_embd=32,
        dropout=0.1,
        device=None
    )

