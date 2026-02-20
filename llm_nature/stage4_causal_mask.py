import argparse
import torch
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def create_causal_mask(seq_len: int):
    return torch.tril(torch.ones(seq_len, seq_len))

def run(ctx: RunContext) -> dict:
    torch.manual_seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 4: CAUSAL MASK — THE ARROW OF TIME"))
    lines.append(friend_sentence(
        4,
        "The model predicts the next token without ...",
        "Without using grammar rules.",
        "Without peeking at future tokens (causal mask blocks them)."
    ))

    seq_len = 8
    mask = create_causal_mask(seq_len)

    fig = plt.figure(figsize=(6,5))
    plt.imshow(mask.numpy(), aspect="auto")
    plt.title("Causal Mask (✓ allowed, ✗ blocked)")
    plt.xlabel("Attend-to position")
    plt.ylabel("Current position")
    for i in range(seq_len):
        for j in range(seq_len):
            plt.text(j, i, "✓" if mask[i,j] > 0 else "✗", ha="center", va="center")
    fig_path = f"{ctx.out_dir}/causal_mask.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close(fig)

    lines.append("\n🔍 Rule: token i attends only to tokens ≤ i.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 4,
        "seed": ctx.seed,
        "figure": "causal_mask.png",
        "mask": mask.int().tolist()
    })
    return {"stage": 4, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
