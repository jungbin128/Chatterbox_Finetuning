import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        h = self.ff(x)
        return self.norm2(x + h)


class T3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["dim"]

        self.text_emb = nn.Embedding(cfg["text_vocab"], d)
        self.text_pos = nn.Embedding(cfg["max_text"], d)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg["heads"]) for _ in range(cfg["layers"])]
        )

        self.sem_emb = nn.Embedding(cfg["speech_vocab"], d)
        self.sem_pos = nn.Embedding(cfg["max_speech"], d)

        self.to_vocab = nn.Linear(d, cfg["speech_vocab"])
        self.cfg = cfg

    def encode_text(self, txt):
        pos = torch.arange(txt.size(1), device=txt.device)
        x = self.text_emb(txt) + self.text_pos(pos)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=1, keepdim=True)

    def forward_logits(self, txt, semantic_target):
        B, T = semantic_target.shape
        device = semantic_target.device

        h = self.encode_text(txt)

        pos = torch.arange(T, device=device)
        y = self.sem_emb(semantic_target) + self.sem_pos(pos)
        y = y + h

        for blk in self.blocks:
            y = blk(y)

        logits = self.to_vocab(y)
        return logits

    @torch.no_grad()
    def generate(self, txt, max_len=1024):
        B = txt.size(0)
        device = txt.device

        h = self.encode_text(txt)

        tokens = []
        cur = torch.zeros(B, 1, dtype=torch.long, device=device)

        for i in range(max_len):
            pos = torch.tensor([i], device=device)
            y = self.sem_emb(cur) + self.sem_pos(pos)
            y = y + h

            for blk in self.blocks:
                y = blk(y)

            logits = self.to_vocab(y[:, -1])
            nxt = logits.argmax(dim=-1, keepdim=True)
            tokens.append(nxt)
            cur = nxt

        return torch.cat(tokens, dim=1)
