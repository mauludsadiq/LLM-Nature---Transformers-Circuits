"""
Stage 15: Complete Picture (simulated)
Writes a single synthesized figure and a structured summary.
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
    out_dir = out_root / "stage15_complete_picture"
    out_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = [
        {"level": 1, "name": "Raw neurons", "tag": "polysemantic"},
        {"level": 2, "name": "Sparse features", "tag": "interpretable"},
        {"level": 3, "name": "Circuits", "tag": "specialized"},
        {"level": 4, "name": "Universal features", "tag": "cross-model"},
        {"level": 5, "name": "Truth geometry", "tag": "directional"},
        {"level": 6, "name": "Steering", "tag": "control"},
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.linspace(0.08, 0.92, len(hierarchy))
    for i, h in enumerate(hierarchy):
        x = xs[i]
        rect = plt.Rectangle((x - 0.07, 0.35), 0.14, 0.3, alpha=0.75)
        ax.add_patch(rect)
        ax.text(x, 0.50, f"L{h['level']}\n{h['name']}", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x, 0.30, h["tag"], ha="center", va="center", fontsize=7)
        if i < len(hierarchy) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.07, 0.5), xytext=(x + 0.07, 0.5), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Stage 15: From Neurons to Truth (simulated)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "complete_picture.png", dpi=160)
    plt.close(fig)

    artifacts = {
        "stage": 15,
        "hierarchy": hierarchy,
        "claim": "This stage is a synthesized map of the pipeline stages as stable output for human inspection.",
    }
    _write_json(out_dir / "artifacts.json", artifacts)

    printed = []
    printed.append("=" * 80)
    printed.append("EXPERIMENT 15: THE COMPLETE PICTURE - FROM NEURONS TO TRUTH (SIMULATED)")
    printed.append("=" * 80)
    printed.append("")
    printed.append("You: The friend finishes my sentence, so the model must be magic.")
    printed.append("Friend: ...and if it is magic, it cannot be decomposed into parts.")
    printed.append("Reality: We pin a hierarchy of artifacts: neurons → features → circuits → universality → geometry → steering.")
    printed.append("")
    printed.append("Artifacts: complete_picture.png  artifacts.json")
    printed.append("")
    _write_text(out_dir / "printed_output.txt", "\n".join(printed) + "\n")

    return artifacts
