import argparse
import torch
import torch.nn.functional as F
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def self_attention(x, mask=None):
    batch_size, seq_len, d_k = x.size()
    q = x
    k = x
    v = x
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output, weights

def run(ctx: RunContext) -> dict:
    torch.manual_seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 1: SELF-ATTENTION FROM SCRATCH"))
    lines.append(friend_sentence(
        1,
        "The cat sat on the ...",
        "The cat sat on the mat. (confident)",
        "At this stage, attention is just weighted mixing — no stored facts yet."
    ))
    lines.append("\n🔤 Testing with tiny word embeddings...\n")

    vocab = {"The": [1.0, 0.0, 0.0, 0.1],
             "cat": [0.0, 1.0, 0.5, 0.0],
             "sat": [0.0, 0.0, 1.0, 0.9]}
    x = torch.tensor([vocab["The"], vocab["cat"], vocab["sat"]]).unsqueeze(0)
    y, w = self_attention(x)

    lines.append("\nAttention Weights (row = word looking, col = word at):\n")
    lines.append("         The    cat    sat\n")
    for i, word in enumerate(["The", "cat", "sat"]):
        row = w[0, i].detach().numpy()
        lines.append(f"{word:8} {row[0]:.2f}   {row[1]:.2f}   {row[2]:.2f}\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 1,
        "seed": ctx.seed,
        "attention_weights": w[0].detach().tolist(),
        "output": y[0].detach().tolist(),
    })
    return {"stage": 1, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
