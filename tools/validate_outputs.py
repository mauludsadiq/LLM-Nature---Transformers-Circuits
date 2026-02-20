from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def validate_stage7(out_dir: Path) -> None:
    p = out_dir / "artifacts.json"
    data = _load(p)

    need = ["clean_prompt", "corrupted_prompt", "top_before", "top_after", "p_paris", "p_london"]
    for k in need:
        if k not in data:
            raise ValueError(f"missing key {k!r} in {p}")

    pb = float(data["p_paris"]["before"])
    pa = float(data["p_paris"]["after"])
    lb = float(data["p_london"]["before"])
    la = float(data["p_london"]["after"])

    if not (pa > pb):
        raise ValueError(f"expected Paris to increase after patch, got before={pb} after={pa}")

    if not (la < lb):
        raise ValueError(f"expected London to decrease after patch, got before={lb} after={la}")


def validate_stage7_sweep(out_dir: Path) -> None:
    p = out_dir / "artifacts.json"
    data = _load(p)

    if data.get("kind") != "sweep":
        raise ValueError("not a sweep artifacts.json")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError("rows missing or empty")

    for r in rows:
        if "layer" not in r:
            raise ValueError("row missing layer")
        pp = r["p_paris"]
        pl = r["p_london"]
        for kk in ["clean", "before", "after"]:
            float(pp[kk])
            float(pl[kk])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage7", default="")
    ap.add_argument("--stage7_sweep", default="")
    args = ap.parse_args()

    if args.stage7:
        validate_stage7(Path(args.stage7))
    if args.stage7_sweep:
        validate_stage7_sweep(Path(args.stage7_sweep))


if __name__ == "__main__":
    main()
