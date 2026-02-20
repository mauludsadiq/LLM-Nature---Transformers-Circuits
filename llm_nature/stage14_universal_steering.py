"""
Stage 14: Universal Feature Steering (simulated)
Encodes transfer effectiveness table as deterministic artifacts.
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
    out_dir = out_root / "stage14_universal_steering"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("GPT-2→Pythia", 0.82),
        ("GPT-2→LLaMA", 0.76),
        ("Pythia→GPT-2", 0.79),
        ("Pythia→LLaMA", 0.81),
        ("LLaMA→GPT-2", 0.74),
        ("LLaMA→Pythia", 0.78),
    ]

    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 3))
    y = np.arange(len(names))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("transfer effectiveness (simulated)")
    ax.set_title("Stage 14: Cross-Model Steering Transfer (simulated)")
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_effectiveness.png", dpi=160)
    plt.close(fig)

    artifacts = {
        "stage": 14,
        "transfer_pairs": [{"pair": n, "effectiveness": float(v)} for n, v in pairs],
        "mean_effectiveness": float(np.mean(vals)),
        "note": "Simulated cross-model steering transfer as a stable test artifact.",
    }
    _write_json(out_dir / "artifacts.json", artifacts)

    printed = []
    printed.append("=" * 80)
    printed.append("EXPERIMENT 14: UNIVERSAL FEATURE STEERING (SIMULATED)")
    printed.append("=" * 80)
    printed.append("")
    printed.append("You: If I learn a knob on one friend, it will not work on another friend.")
    printed.append("Friend: ...but if the features are universal, the same knob works everywhere.")
    printed.append("Reality: We encode a transfer table and validate its deterministic mean.")
    printed.append("")
    printed.append(f"Mean transfer effectiveness: {artifacts['mean_effectiveness']:.3f}")
    printed.append("Artifacts: transfer_effectiveness.png  artifacts.json")
    printed.append("")
    _write_text(out_dir / "printed_output.txt", "\n".join(printed) + "\n")

    return artifacts
