import argparse
import numpy as np
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def run(ctx: RunContext) -> dict:
    np.random.seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 10: COMPLETE PIPELINE (MAP)"))
    lines.append(friend_sentence(
        10,
        "Interpretability is just pictures because ...",
        "Because we can only look, not touch.",
        "We can measure and intervene (patch/edit) without retraining."
    ))

    fig, ax = plt.subplots(figsize=(12,4))
    steps = ["Load GPT-2", "Logit lens", "Circuit discovery", "Causal tracing", "SAE features", "Intervention"]
    x = np.arange(len(steps))
    ax.scatter(x, np.ones_like(x))
    for i, s in enumerate(steps):
        ax.text(i, 1.05, s, ha="center", fontsize=9)
        if i < len(steps)-1:
            ax.annotate("", xy=(i+1, 1), xytext=(i, 1), arrowprops=dict(arrowstyle="->"))
    ax.set_ylim(0.8, 1.2)
    ax.axis("off")
    ax.set_title("Pipeline overview: observation → circuit → intervention")
    fig1 = f"{ctx.out_dir}/pipeline_overview.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close(fig)

    lines.append("\n🚀 Outcome: complete workflow from first principles to targeted edits.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 10,
        "seed": ctx.seed,
        "figure": "pipeline_overview.png",
        "steps": steps
    })
    return {"stage": 10, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
