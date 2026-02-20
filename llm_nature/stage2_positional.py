import argparse, math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def run(ctx: RunContext) -> dict:
    torch.manual_seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 2: POSITIONAL ENCODINGS"))
    lines.append(friend_sentence(
        2,
        "Order matters in a sentence because ...",
        "Because the model reads left-to-right only.",
        "Transformers are set-like; positions are stamped in via encodings."
    ))

    pe = PositionalEncoding(d_model=16)
    positions = torch.arange(0, 50).unsqueeze(0).unsqueeze(-1).expand(-1, -1, 16).float()
    encoded = pe(positions)

    fig = plt.figure(figsize=(12, 4))
    plt.imshow(encoded[0, :, :8].detach().numpy().T, aspect="auto")
    plt.xlabel("Position")
    plt.ylabel("Encoding Dimension")
    plt.title("Sinusoidal Positional Encodings (first 8 dims)")
    plt.colorbar(label="Value")
    fig_path = f"{ctx.out_dir}/positional_encodings.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close(fig)

    lines.append("\n✅ Positional encodings give each token a unique 'address'.\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 2,
        "seed": ctx.seed,
        "figure": "positional_encodings.png",
        "encoded_shape": list(encoded.shape),
        "encoded_sample": encoded[0, :5, :8].detach().tolist()
    })
    return {"stage": 2, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
