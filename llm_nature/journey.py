from . import stage15_complete_picture as stage15_complete_picture
from . import stage14_universal_steering as stage14_universal_steering
from . import stage13_truth_geometry as stage13_truth_geometry
from . import stage12_real_world_evidence as stage12_real_world_evidence
from . import stage11_universal_features as stage11_universal_features
import argparse, os
from .util import RunContext, ensure_dir, write_json, banner
from . import stage1_attention, stage2_positional, stage3_block, stage4_causal_mask, stage5_logit_lens
from . import stage6_ioi_circuit, stage7_patching, stage8_sae, stage9_vsa, stage10_pipeline

STAGES = [
    ("stage1", stage1_attention.run),
    ("stage2", stage2_positional.run),
    ("stage3", stage3_block.run),
    ("stage4", stage4_causal_mask.run),
    ("stage5", stage5_logit_lens.run),
    ("stage6", stage6_ioi_circuit.run),
    ("stage7", stage7_patching.run),
    ("stage8", stage8_sae.run),
    ("stage9", stage9_vsa.run),
    ("stage10", stage10_pipeline.run),
]

def run_all(out_dir: str, seed: int) -> dict:
    ensure_dir(out_dir)
    printed_master = []
    for name, fn in STAGES:
        sub = os.path.join(out_dir, name)
        r = fn(RunContext(out_dir=sub, seed=seed))
        printed_master.append(banner(f"STAGE OUTPUT: {name}"))
        printed_master.append(r.get("printed", ""))
        printed_master.append("\n")
    master = "".join(printed_master)
    with open(os.path.join(out_dir, "PRINTED_OUTPUT_ALL_STAGES.txt"), "w", encoding="utf-8") as f:
        f.write(master)
    return {"out_dir": out_dir, "seed": seed, "stages": [n for n,_ in STAGES]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    summary = run_all(a.out, a.seed)
    write_json(os.path.join(a.out, "run_summary.json"), summary)

if __name__ == "__main__":
    main()
