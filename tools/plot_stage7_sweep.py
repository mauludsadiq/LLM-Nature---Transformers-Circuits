from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(p: Path):
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    for row in rows:
        row["layer"] = int(row["layer"])
        for k in [
            "delta_paris",
            "delta_london",
            "delta_logodds_paris",
            "delta_logodds_london",
        ]:
            row[k] = float(row[k])
    rows.sort(key=lambda x: x["layer"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    layers = [r["layer"] for r in rows]

    dlp = [r["delta_logodds_paris"] for r in rows]
    dll = [r["delta_logodds_london"] for r in rows]
    dp = [r["delta_paris"] for r in rows]
    dl = [r["delta_london"] for r in rows]

    plt.figure()
    plt.plot(layers, dlp, marker="o")
    plt.plot(layers, dll, marker="o")
    plt.xlabel("layer")
    plt.ylabel("delta log-odds (after - before)")
    plt.title("Stage 7 sweep: log-odds shifts")
    plt.legend(["Paris", "London"], loc="best")
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    plt.close()

    out2 = str(Path(args.out).with_suffix(".deltas.png"))
    plt.figure()
    plt.plot(layers, dp, marker="o")
    plt.plot(layers, dl, marker="o")
    plt.xlabel("layer")
    plt.ylabel("delta p (after - before)")
    plt.title("Stage 7 sweep: probability deltas")
    plt.legend(["Paris", "London"], loc="best")
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
