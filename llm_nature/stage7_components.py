from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .util import RunContext, ensure_dir, write_json, write_text


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


def topk(tok: AutoTokenizer, probs: torch.Tensor, k: int = 10) -> List[Dict[str, float]]:
    vals, idx = torch.topk(probs, k)
    out = []
    for p, i in zip(vals.tolist(), idx.tolist()):
        out.append({"token": tok.decode([int(i)]), "token_id": int(i), "p": float(p)})
    return out


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


def _pick_module(model: AutoModelForCausalLM, layer: int, component: str) -> Any:
    block = model.transformer.h[layer]
    if component == "block":
        return block
    if component == "attn":
        return block.attn
    if component == "mlp":
        return block.mlp
    raise ValueError(f"unknown component {component!r}")


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

    layer = int(os.environ.get("LLM_NATURE_LAYER", "11"))
    component = os.environ.get("LLM_NATURE_COMPONENT", "block")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    id_paris = _tok_id(tok, token_paris)
    id_london = _tok_id(tok, token_london)

    logits_clean, ids_clean = _run_logits(model, tok, clean, device)
    probs_clean = _softmax_last(logits_clean)

    pos = int(os.environ.get("LLM_NATURE_PATCH_POS", str(len(ids_clean) - 1)))

    clean_vec: Optional[torch.Tensor] = None

    def grab(module: Any, inp: Any, out: Any) -> None:
        nonlocal clean_vec
        hs = out[0] if isinstance(out, tuple) else out
        clean_vec = hs[:, pos, :].detach().clone()

    mod_clean = _pick_module(model, layer, component)
    h_clean = mod_clean.register_forward_hook(grab)
    _run_logits(model, tok, clean, device)
    h_clean.remove()

    if clean_vec is None:
        raise RuntimeError("failed to capture clean vector")

    logits_before, _ = _run_logits(model, tok, corrupted, device)
    probs_before = _softmax_last(logits_before)

    mod_patch = _pick_module(model, layer, component)
    h_patch = mod_patch.register_forward_hook(PatchHook(clean_vec, pos))
    logits_after, _ = _run_logits(model, tok, corrupted, device)
    h_patch.remove()

    probs_after = _softmax_last(logits_after)

    top_clean = topk(tok, probs_clean, 10)
    top_before = topk(tok, probs_before, 10)
    top_after = topk(tok, probs_after, 10)

    p_paris = {
        "clean": float(probs_clean[id_paris].item()),
        "before": float(probs_before[id_paris].item()),
        "after": float(probs_after[id_paris].item()),
    }
    p_london = {
        "clean": float(probs_clean[id_london].item()),
        "before": float(probs_before[id_london].item()),
        "after": float(probs_after[id_london].item()),
    }

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT 7 COMPONENT PATCH")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Device: {device}")
    lines.append(f"Model: {model_name}")
    lines.append(f"Layer: {layer}")
    lines.append(f"Component: {component}")
    lines.append(f"Patch pos: {pos}")
    lines.append("")
    lines.append(f"Clean prompt: {clean!r}")
    lines.append("Top-10 next tokens (CLEAN):")
    for t in top_clean:
        lines.append(f"   {t['token']!r:10} p={t['p']:.4f}")
    lines.append("")
    lines.append(f"Corrupted prompt: {corrupted!r}")
    lines.append("Top-10 next tokens BEFORE patch:")
    for t in top_before:
        lines.append(f"   {t['token']!r:10} p={t['p']:.4f}")
    lines.append("")
    lines.append("Top-10 next tokens AFTER patch:")
    for t in top_after:
        lines.append(f"   {t['token']!r:10} p={t['p']:.4f}")
    lines.append("")
    lines.append("Target token probabilities:")
    lines.append(f"   p({token_paris!r})  clean={p_paris['clean']}  before={p_paris['before']}  after={p_paris['after']}")
    lines.append(f"   p({token_london!r}) clean={p_london['clean']} before={p_london['before']} after={p_london['after']}")
    lines.append("")

    write_text(str(out_dir / "printed_output.txt"), "\n".join(lines) + "\n")

    artifacts: Dict[str, Any] = {
        "stage": 7,
        "kind": "component",
        "seed": seed,
        "device": device,
        "model": model_name,
        "clean_prompt": clean,
        "corrupted_prompt": corrupted,
        "layer": layer,
        "component": component,
        "patch_pos": pos,
        "top_clean": top_clean,
        "top_before": top_before,
        "top_after": top_after,
        "p_paris": p_paris,
        "p_london": p_london,
    }
    write_json(str(out_dir / "artifacts.json"), artifacts)

    return {"stage": 7, "kind": "component", "layer": layer, "component": component}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(RunContext(out_dir=args.out, seed=args.seed))


if __name__ == "__main__":
    main()
