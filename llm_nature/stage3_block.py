import argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        b, t, _ = x.size()
        Q = self.W_Q(x).view(b, t, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(x).view(b, t, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(x).view(b, t, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        w = F.softmax(scores, dim=-1)
        ctx = torch.matmul(w, V)
        ctx = ctx.transpose(1,2).contiguous().view(b, t, self.d_model)
        return self.W_O(ctx), w

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        a, w = self.attn(x, mask)
        x = self.norm1(x + a)
        f = self.ff(x)
        x = self.norm2(x + f)
        return x, w

def run(ctx: RunContext) -> dict:
    torch.manual_seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 3: COMPLETE TRANSFORMER BLOCK"))
    lines.append(friend_sentence(
        3,
        "A transformer block is attention plus ...",
        "Plus more attention.",
        "Attention + MLP, each wrapped in residual + layernorm."
    ))
    lines.append("✅ Built transformer block: MHA + residual + layernorm + FFN.\n")

    x = torch.randn(1, 6, 32)
    block = TransformerBlock(d_model=32, n_heads=4, d_ff=64)
    y, w = block(x)

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 3,
        "seed": ctx.seed,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "attn_weights_shape": list(w.shape),
    })
    return {"stage": 3, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
