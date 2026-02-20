"""
Stage 13: Geometry of Truth (simulated)
Produces a truth direction and shows separation in projection space.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(out_root: str | Path) -> dict:
    out_root = Path(out_root)
    out_dir = out_root / "stage13_truth_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n = 80

    truth_dir = np.array([1.0, 0.5, 0.3], dtype=np.float64)
    truth_dir = truth_dir / np.linalg.norm(truth_dir)

    true_pts = np.random.randn(n, 3)
    true_pts = true_pts / np.linalg.norm(true_pts, axis=1, keepdims=True)
    true_pts = true_pts + 0.8 * truth_dir

    false_pts = np.random.randn(n, 3)
    false_pts = false_pts / np.linalg.norm(false_pts, axis=1, keepdims=True)
    false_pts = false_pts - 0.8 * truth_dir

    true_proj = (true_pts @ truth_dir).astype(np.float64)
    false_proj = (false_pts @ truth_dir).astype(np.float64)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(true_proj, bins=20, alpha=0.7, label="True", color="green")
    ax.hist(false_proj, bins=20, alpha=0.7, label="False", color="red")
    ax.axvline(x=0.0, color="black", linestyle="--")
    ax.set_title("Stage 13: Projection onto Truth Direction (simulated)")
    ax.set_xlabel("projection")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "truth_projection_hist.png", dpi=160)
    plt.close(fig)

    artifacts = {
        "stage": 13,
        "truth_direction": truth_dir.tolist(),
        "true_projection_mean": float(true_proj.mean()),
        "false_projection_mean": float(false_proj.mean()),
        "separation": float(true_proj.mean() - false_proj.mean()),
        "n_points": int(n),
    }
    _write_json(out_dir / "artifacts.json", artifacts)

    printed = []
    printed.append("=" * 80)
    printed.append("EXPERIMENT 13: THE GEOMETRY OF TRUTH (SIMULATED)")
    printed.append("=" * 80)
    printed.append("")
    printed.append("You: The friend finishes my sentence with confidence, even when wrong.")
    printed.append("Friend: ...so truth must be a vibe, not a measurable quantity.")
    printed.append("Reality: We define a direction and measure separation by projection statistics.")
    printed.append("")
    printed.append(f"True mean projection: {artifacts['true_projection_mean']:.3f}")
    printed.append(f"False mean projection: {artifacts['false_projection_mean']:.3f}")
    printed.append(f"Mean separation: {artifacts['separation']:.3f}")
    printed.append("Artifacts: truth_projection_hist.png  artifacts.json")
    printed.append("")
    _write_text(out_dir / "printed_output.txt", "\n".join(printed) + "\n")

    return artifacts
