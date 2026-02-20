import argparse
import numpy as np
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def run(ctx: RunContext) -> dict:
    np.random.seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 9: VECTOR SYMBOLIC ARCHITECTURE (VSA) VIEW"))
    lines.append(friend_sentence(
        9,
        "High-dimensional vectors interfere constantly because ...",
        "Because the space is crowded.",
        "Random vectors become nearly orthogonal as dimension grows."
    ))

    dims = [10, 50, 100, 500, 1000, 5000]
    mean_dots = []
    for dim in dims:
        v = np.random.randn(500, dim)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        dots = []
        for i in range(100):
            for j in range(i+1, 101):
                dots.append(abs(np.dot(v[i], v[j])))
        mean_dots.append(float(np.mean(dots)))

    fig = plt.figure(figsize=(8,4))
    plt.plot(dims, mean_dots, marker="o")
    plt.xscale("log")
    plt.xlabel("Dimension (log)")
    plt.ylabel("Mean |dot|")
    plt.title("Near-orthogonality in high dimensions")
    plt.grid(True, alpha=0.3)
    fig1 = f"{ctx.out_dir}/vsa_orthogonality.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close(fig)

    lines.append("\n🧠 VSA takeaway: residual stream ≈ concept bundle bus.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 9,
        "seed": ctx.seed,
        "figure": "vsa_orthogonality.png",
        "dims": dims,
        "mean_abs_dot": mean_dots
    })
    return {"stage": 9, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
