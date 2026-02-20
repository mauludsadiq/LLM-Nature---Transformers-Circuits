"""
Stage 11: Universal Features Hypothesis
Simulated cross-model feature alignment with deterministic artifacts and prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class FeatureSimulator:
    n_concepts: int = 20
    n_features: int = 100
    d_model: int = 256

    def __post_init__(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.true_concepts = F.normalize(torch.randn(self.n_concepts, self.d_model), dim=-1)
        names = [
            "Capital cities", "Past tense verbs", "Mathematical operators",
            "Animal names", "Emotional language", "Scientific terms",
            "Legal jargon", "Medical terminology", "Programming keywords",
            "Geographic locations", "Historical figures", "Chemical elements",
            "Musical terms", "Artistic styles", "Religious concepts",
            "Philosophical ideas", "Political ideologies", "Economic terms",
            "Sports terminology", "Food and cuisine", "Fashion terms",
            "Architectural styles", "Military terms", "Space and astronomy",
            "Weather phenomena", "Color concepts", "Time expressions",
            "Family relationships", "Professional roles", "Educational terms"
        ]
        self.concept_names = names[: self.n_concepts]

    def train_model_features(self, model_seed: int, noise_level: float, missing_concepts: int) -> tuple[torch.Tensor, torch.Tensor]:
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)

        n_learned = max(1, self.n_features - missing_concepts)
        learned = torch.randn(n_learned, self.d_model)
        comp = torch.zeros(n_learned, self.n_concepts)

        for i in range(n_learned):
            k = int(np.random.randint(1, 4))
            idxs = np.random.choice(self.n_concepts, k, replace=False)
            w = torch.rand(k)
            w = w / w.sum()

            comp[i, torch.tensor(idxs)] = w
            feat = torch.zeros(self.d_model)
            for j, ww in zip(idxs, w):
                feat = feat + ww * self.true_concepts[int(j)]
            noise = torch.randn(self.d_model) * float(noise_level)
            learned[i] = feat + noise

        learned = F.normalize(learned, dim=-1)
        return learned, comp

    @staticmethod
    def compute_feature_similarity(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
        sim = features_a @ features_b.T
        best = sim.max(dim=1)[0]
        return float(best.mean().item())

    def run(self, out_dir: Path) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)

        n_models = 5
        all_features = []
        all_comp = []

        for seed in range(n_models):
            feats, comp = self.train_model_features(model_seed=seed, noise_level=0.2, missing_concepts=seed)
            all_features.append(feats)
            all_comp.append(comp)

        sim_mat = np.zeros((n_models, n_models), dtype=np.float64)
        for i in range(n_models):
            for j in range(n_models):
                sim_mat[i, j] = self.compute_feature_similarity(all_features[i], all_features[j])

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(sim_mat, vmin=0.0, vmax=1.0)
        ax.set_title("Stage 11: Feature Similarity Across Models")
        ax.set_xlabel("Model")
        ax.set_ylabel("Model")
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        fig.colorbar(im, ax=ax, label="mean best-match cosine")
        for i in range(n_models):
            for j in range(n_models):
                ax.text(j, i, f"{sim_mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "similarity_matrix.png", dpi=160)
        plt.close(fig)

        comp0 = all_comp[0][:20, : min(10, self.n_concepts)].detach().cpu().numpy().tolist()
        comp1 = all_comp[1][:20, : min(10, self.n_concepts)].detach().cpu().numpy().tolist()

        artifacts = {
            "stage": 11,
            "n_models": n_models,
            "n_concepts": self.n_concepts,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "concept_names": self.concept_names,
            "similarity_matrix": sim_mat.tolist(),
            "example_composition_model0_top20_x_top10": comp0,
            "example_composition_model1_top20_x_top10": comp1,
            "avg_cross_model_similarity_model0_model1": float(sim_mat[0, 1]),
        }
        _write_json(out_dir / "artifacts.json", artifacts)

        printed = []
        printed.append("=" * 80)
        printed.append("EXPERIMENT 11: UNIVERSAL FEATURES - DO ALL MODELS LEARN THE SAME TRUTH?")
        printed.append("=" * 80)
        printed.append("")
        printed.append("You: The model is like a friend who tries to finish your sentence...")
        printed.append("Friend: ...so if I train five friends, they will all finish it the same way.")
        printed.append("Reality: We test whether independently trained feature sets align by best-match cosine similarity.")
        printed.append("")
        printed.append(f"Models: {n_models}")
        printed.append(f"Concepts: {self.n_concepts}   Features per model (target): {self.n_features}")
        printed.append(f"Avg best-match similarity (Model 0 → Model 1): {float(sim_mat[0, 1]):.3f}")
        printed.append("Artifact: similarity_matrix.png")
        printed.append("Artifact: artifacts.json")
        printed.append("")
        _write_text(out_dir / "printed_output.txt", "\n".join(printed) + "\n")

        return artifacts


def run(out_root: str | Path) -> dict:
    out_root = Path(out_root)
    stage_dir = out_root / "stage11_universal_features"
    sim = FeatureSimulator(n_concepts=20, n_features=100, d_model=256)
    return sim.run(stage_dir)
