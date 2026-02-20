from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .util import RunContext, ensure_dir, write_json, write_text


@dataclass
class SweepRow:
    layer: int
    pos: int
    p_paris_clean: float
    p_paris_before: float
    p_paris_after: float
    p_london_clean: float
    p_london_before: float
    p_london_after: float

    @staticmethod
    def header() -> List[str]:
        return [
            "layer",
            "pos",
            "p_paris_clean",
            "p_paris_before",
            "p_paris_after",
            "p_london_clean",
            "p_london_before",
            "p_london_after",
            "delta_paris",
            "delta_london",
            "delta_logodds_paris",
            "delta_logodds_london",
        ]

    def to_csv_row(self) -> List[str]:
        def logit(p: float) -> float:
            eps = 1e-12
            p2 = min(1.0 - eps, max(eps, p))
            return math.log(p2 / (1.0 - p2))

        delta_paris = self.p_paris_after - self.p_paris_before
        delta_london = self.p_london_after - self.p_london_before
        dlp = logit(self.p_paris_after) - logit(self.p_paris_before)
        dll = logit(self.p_london_after) - logit(self.p_london_before)

        return [
            str(self.layer),
            str(self.pos),
            repr(self.p_paris_clean),
            repr(self.p_paris_before),
            repr(self.p_paris_after),
            repr(self.p_london_clean),
            repr(self.p_london_before),
            repr(self.p_london_after),
            repr(delta_paris),
            repr(delta_london),
            repr(dlp),
            repr(dll),
        ]


def _tok_id(tok: AutoTokenizer, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"token string must map to one token: {s!r} -> {ids}")
    return int(ids[0])


def _softmax_last(logits: torch.Tensor) -> torch.Tensor:
    v = logits[0, -1, :]
    return torch.softmax(v, dim=-1)


def _run_logits(model: AutoModelForCausalLM, tok: AutoTokenizer, prompt: str, device: str) -> Tuple[torch.Tensor, List[int]]:
    inp = tok(prompt, return_tensors="pt")
    input_ids = inp["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits.detach().cpu(), input_ids[0].detach().cpu().tolist()


class PatchHook:
    def __init__(self, vec: torch.Tensor, pos: int):
        self.vec = vec
        self.pos = pos

    def __call__(self, module: Any, inp: Any, out: Any) -> Any:
        if isinstance(out, tuple):
            hs = out[0]
            rest = out[1:]
        else:
            hs = out
            rest = None

        hs2 = hs.clone()
        hs2[:, self.pos, :] = self.vec.to(hs2.device)

        if rest is None:
            return hs2
        return (hs2,) + rest


def run(ctx: RunContext) -> Dict[str, Any]:
    out_dir = Path(ctx.out_dir)
    ensure_dir(out_dir)

    seed = int(ctx.seed)
    torch.manual_seed(seed)

    device = os.environ.get("LLM_NATURE_DEVICE", "cpu")
    model_name = os.environ.get("LLM_NATURE_MODEL", "gpt2")

    clean = os.environ.get("LLM_NATURE_CLEAN_PROMPT", "The capital of France is")
    corrupted = os.environ.get("LLM_NATURE_CORRUPT_PROMPT", "The capital of England is")

    token_paris = os.environ.get("LLM_NATURE_TOKEN_PARIS", " Paris")
    token_london = os.environ.get("LLM_NATURE_TOKEN_LONDON", " London")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    id_paris = _tok_id(tok, token_paris)
    id_london = _tok_id(tok, token_london)

    logits_clean, ids_clean = _run_logits(model, tok, clean, device)
    probs_clean = _softmax_last(logits_clean)

    p_paris_clean = float(probs_clean[id_paris].item())
    p_london_clean = float(probs_clean[id_london].item())

    pos = int(os.environ.get("LLM_NATURE_PATCH_POS", str(len(ids_clean) - 1)))

    layers_env = os.environ.get("LLM_NATURE_LAYERS", "0,1,2,3,4,5,6,7,8,9,10,11")
    layers = [int(x.strip()) for x in layers_env.split(",") if x.strip() != ""]

    rows: List[SweepRow] = []
    printed: List[str] = []
    printed.append("=" * 80)
    printed.append("EXPERIMENT 7 SWEEP: ACTIVATION PATCHING LAYER SCAN")
    printed.append("=" * 80)
    printed.append("")
    printed.append(f"Device: {device}")
    printed.append(f"Model: {model_name}")
    printed.append(f"Clean prompt: {clean!r}")
    printed.append(f"Corrupted prompt: {corrupted!r}")
    printed.append(f"Patch pos: {pos}")
    printed.append(f"Target token Paris: {token_paris!r} id={id_paris}")
    printed.append(f"Target token London: {token_london!r} id={id_london}")
    printed.append("")

    for layer in layers:
        clean_vec: Optional[torch.Tensor] = None

        def grab_clean(module: Any, inp: Any, out: Any) -> None:
            nonlocal clean_vec
            hs = out[0] if isinstance(out, tuple) else out
            clean_vec = hs[:, pos, :].detach().clone()

        h_clean = model.transformer.h[layer].register_forward_hook(grab_clean)
        _run_logits(model, tok, clean, device)
        h_clean.remove()

        if clean_vec is None:
            raise RuntimeError(f"failed to capture clean vector at layer {layer}")

        logits_before, _ = _run_logits(model, tok, corrupted, device)
        probs_before = _softmax_last(logits_before)
        p_paris_before = float(probs_before[id_paris].item())
        p_london_before = float(probs_before[id_london].item())

        hook = PatchHook(vec=clean_vec, pos=pos)
        h_patch = model.transformer.h[layer].register_forward_hook(hook)
        logits_after, _ = _run_logits(model, tok, corrupted, device)
        h_patch.remove()

        probs_after = _softmax_last(logits_after)
        p_paris_after = float(probs_after[id_paris].item())
        p_london_after = float(probs_after[id_london].item())

        row = SweepRow(
            layer=layer,
            pos=pos,
            p_paris_clean=p_paris_clean,
            p_paris_before=p_paris_before,
            p_paris_after=p_paris_after,
            p_london_clean=p_london_clean,
            p_london_before=p_london_before,
            p_london_after=p_london_after,
        )
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r.p_london_before - r.p_london_after), reverse=True)

    printed.append("Top layers by London decrease (before - after):")
    for r in rows_sorted[:8]:
        dec = r.p_london_before - r.p_london_after
        inc = r.p_paris_after - r.p_paris_before
        printed.append(f"layer={r.layer:2d}  dLondon={dec:.6f}  dParis={inc:.6f}  Paris(after)={r.p_paris_after:.6f}  London(after)={r.p_london_after:.6f}")

    printed.append("")
    printed_path = out_dir / "printed_output.txt"
    write_text(str(printed_path), "\n".join(printed) + "\n")

    artifacts: Dict[str, Any] = {
        "stage": 7,
        "kind": "sweep",
        "seed": seed,
        "device": device,
        "model": model_name,
        "clean_prompt": clean,
        "corrupted_prompt": corrupted,
        "patch_pos": pos,
        "token_paris": {"text": token_paris, "id": id_paris},
        "token_london": {"text": token_london, "id": id_london},
        "rows": [
            {
                "layer": r.layer,
                "pos": r.pos,
                "p_paris": {"clean": r.p_paris_clean, "before": r.p_paris_before, "after": r.p_paris_after},
                "p_london": {"clean": r.p_london_clean, "before": r.p_london_before, "after": r.p_london_after},
            }
            for r in rows
        ],
    }
    write_json(str(out_dir / "artifacts.json"), artifacts)

    csv_path = out_dir / "stage7_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SweepRow.header())
        for r in rows:
            w.writerow(r.to_csv_row())

    return {"stage": 7, "kind": "sweep", "n_rows": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(RunContext(out_dir=args.out, seed=args.seed))


if __name__ == "__main__":
    main()
