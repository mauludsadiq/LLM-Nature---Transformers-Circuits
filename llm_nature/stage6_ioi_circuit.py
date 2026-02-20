import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .util import RunContext, banner, friend_sentence, write_text, write_json, ensure_dir

def run(ctx: RunContext) -> dict:
    np.random.seed(ctx.seed)
    ensure_dir(ctx.out_dir)

    lines = []
    lines.append(banner("EXPERIMENT 6: IOI CIRCUIT — SPECIALIZED HEADS"))
    lines.append(friend_sentence(
        6,
        "The model resolves 'gave it to ?' by ...",
        "By picking the most recent name.",
        "By a circuit: duplicate-detect → inhibit → name-move."
    ))

    G = nx.DiGraph()
    nodes = ["Previous Token", "Duplicate Detection", "S-Inhibition", "Name Mover", "Output"]
    G.add_nodes_from(nodes)
    G.add_edges_from([
        ("Previous Token","Duplicate Detection"),
        ("Duplicate Detection","S-Inhibition"),
        ("S-Inhibition","Name Mover"),
        ("Name Mover","Output")
    ])

    fig = plt.figure(figsize=(10,4))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw(G, pos, with_labels=True, node_size=2000, font_size=9, arrows=True)
    fig1 = f"{ctx.out_dir}/ioi_circuit_graph.png"
    plt.tight_layout(); plt.savefig(fig1, dpi=160); plt.close(fig)

    layers = np.zeros((12,12), dtype=int)
    for layer in [4,5]: layers[layer,2:5]=1
    for layer in [7,8,9,10]: layers[layer,3:8]=2
    for layer in [7,8]: layers[layer,0:2]=3
    for layer in [9,10,11]: layers[layer,6:11]=4

    fig = plt.figure(figsize=(8,4))
    plt.imshow(layers.T, aspect="auto", origin="lower")
    plt.xlabel("Layer"); plt.ylabel("Head")
    plt.title("IOI head types across layers (toy layout)")
    fig2 = f"{ctx.out_dir}/ioi_head_distribution.png"
    plt.tight_layout(); plt.savefig(fig2, dpi=160); plt.close(fig)

    lines.append("\n🔬 Takeaway: behavior is distributed across heads (a circuit).\n")

    txt = "".join(lines)
    write_text(f"{ctx.out_dir}/printed_output.txt", txt)
    write_json(f"{ctx.out_dir}/artifacts.json", {
        "stage": 6,
        "seed": ctx.seed,
        "figures": ["ioi_circuit_graph.png", "ioi_head_distribution.png"],
        "layers_matrix": layers.tolist()
    })
    return {"stage": 6, "printed": txt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(RunContext(out_dir=a.out, seed=a.seed))

if __name__ == "__main__":
    main()
