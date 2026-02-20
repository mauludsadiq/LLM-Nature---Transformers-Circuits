import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, n_features, sparsity_lambda=0.05):
        super().__init__()
        self.encoder = nn.Linear(d_model, n_features)
        self.decoder = nn.Linear(n_features, d_model)
        self.lmb = sparsity_lambda
    def forward(self, x):
        feats = torch.relu(self.encoder(x))
        sp = self.lmb * feats.abs().mean()
        rec = self.decoder(feats)
        return feats, rec, sp

def run(ctx: RunContext) -> dict:
    torch.manual_seed(ctx.seed)
    np.random.seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 8: SPARSE AUTOENCODERS — FINDING FEATURES"))
    lines.append(friend_sentence(
        8,
        "One neuron means one concept because ...",
        "Because that's intuitive.",
        "Transformers use superposition; SAEs recover sparse features."
    ))

    n_neurons, n_concepts = 10, 20
    neuron_concept = (np.random.rand(n_neurons, n_concepts) > 0.7).astype(int)

    fig = plt.figure(figsize=(10,4))
    plt.imshow(neuron_concept, aspect="auto")
    plt.xlabel("Concept"); plt.ylabel("Neuron")
    plt.title("Toy polysemantic neuron-concept responses")
    fig1 = f"{ctx.out_dir}/polysemantic_matrix.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close(fig)

    d_model = 32
    sae = SparseAutoencoder(d_model=d_model, n_features=64, sparsity_lambda=0.05)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    x = torch.randn(256, d_model)
    for _ in range(200):
        feats, rec, sp = sae(x)
        loss = ((rec - x)**2).mean() + sp
        opt.zero_grad(); loss.backward(); opt.step()

    feats, rec, sp = sae(x)
    plot_feats = feats.detach().numpy()[:50, :40].T

    fig = plt.figure(figsize=(10,5))
    plt.imshow(plot_feats, aspect="auto")
    plt.xlabel("Sample"); plt.ylabel("Feature")
    plt.title("SAE latent activations (toy)")
    fig2 = f"{ctx.out_dir}/sae_features.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=160); plt.close(fig)

    lines.append("\n🧩 SAE takeaway: extract sparse features from a dense stream.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 8,
        "seed": ctx.seed,
        "figures": ["polysemantic_matrix.png", "sae_features.png"],
        "final_loss": float(loss.detach().cpu().item()),
        "final_sparsity": float(sp.detach().cpu().item()),
    })
    return {"stage": 8, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
