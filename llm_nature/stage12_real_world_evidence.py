"""
Stage 12: Real-World Evidence (simulated)
Writes deterministic artifacts that mirror cross-architecture feature universality claims.
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
    out_dir = out_root / "stage12_real_world_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = ["Capital Cities", "Past Tense", "Math Ops", "Animals", "Emotions"]
    gpt2 = [0.85, 0.82, 0.79, 0.88, 0.81]
    pythia = [0.82, 0.79, 0.83, 0.85, 0.78]
    llama = [0.88, 0.84, 0.86, 0.87, 0.83]

    sim = np.array(
        [
            [1.00, 0.76, 0.72],
            [0.76, 1.00, 0.79],
            [0.72, 0.79, 1.00],
        ],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(categories))
    w = 0.25
    axes[0].bar(x - w, gpt2, w, label="GPT-2")
    axes[0].bar(x, pythia, w, label="Pythia")
    axes[0].bar(x + w, llama, w, label="LLaMA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Stage 12: Feature Strength Consistency (simulated)")
    axes[0].legend(fontsize=8)

    im = axes[1].imshow(sim, vmin=0.5, vmax=1.0)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_xticklabels(["GPT-2", "Pythia", "LLaMA"], fontsize=8)
    axes[1].set_yticklabels(["GPT-2", "Pythia", "LLaMA"], fontsize=8)
    axes[1].set_title("Stage 12: Feature Space Similarity (simulated)")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "real_world_evidence.png", dpi=160)
    plt.close(fig)

    printed = []
    printed.append("=" * 80)
    printed.append("EXPERIMENT 12: REAL-WORLD EVIDENCE - GPT-2 vs PYTHIA (SIMULATED)")
    printed.append("=" * 80)
    printed.append("")
    printed.append("You: If the friend finishes my sentence wrong, another friend trained elsewhere will disagree.")
    printed.append("Friend: ...but if they learned the same concepts, they will still finish it the same way.")
    printed.append("Reality: We record consistent feature categories and a cross-model similarity matrix.")
    printed.append("")
    printed.append("Artifacts: real_world_evidence.png  artifacts.json")
    printed.append("")
    _write_text(out_dir / "printed_output.txt", "\n".join(printed) + "\n")

    artifacts = {
        "stage": 12,
        "categories": categories,
        "feature_strengths": {"gpt2": gpt2, "pythia": pythia, "llama": llama},
        "similarity_matrix": sim.tolist(),
        "note": "Simulated summary of cross-architecture universality claims.",
    }
    _write_json(out_dir / "artifacts.json", artifacts)
    return artifacts
