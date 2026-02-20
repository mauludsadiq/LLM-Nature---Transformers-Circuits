import argparse
import numpy as np
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def run(ctx: RunContext) -> dict:
    np.random.seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 5: LOGIT LENS — WATCHING THE MODEL THINK"))
    lines.append(friend_sentence(
        5,
        "The model 'decides' immediately because ...",
        "Because it’s a lookup table.",
        "Predictions sharpen through layers; logit lens shows the curve."
    ))

    layers = 12
    correct = "Paris"
    alts = ["London","Berlin","Rome","Madrid"]

    preds = []
    for l in range(layers):
        if l < 3: p = [0.2]*5
        elif l < 6: p = [0.4,0.3,0.15,0.1,0.05]
        elif l < 9: p = [0.7,0.15,0.08,0.05,0.02]
        else: p = [0.9,0.05,0.03,0.01,0.01]
        preds.append(p)
    preds = np.array(preds)

    fig = plt.figure(figsize=(12,6))
    for i, tok in enumerate([correct]+alts):
        plt.plot(range(layers), preds[:,i], label=tok, linewidth=2 if i==0 else 1, alpha=0.9 if i==0 else 0.5)
    plt.xlabel("Layer"); plt.ylabel("Probability")
    plt.title("Simulated Logit Lens: prediction sharpening through depth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = f"{ctx.out_dir}/logit_lens.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close(fig)

    lines.append("\n🧠 Interpretation: early=surface, middle=candidates, late=commit.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 5,
        "seed": ctx.seed,
        "figure": "logit_lens.png",
        "tokens": [correct]+alts,
        "probs": preds.tolist()
    })
    return {"stage": 5, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
