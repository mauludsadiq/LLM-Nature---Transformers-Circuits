#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-_out}"
python -m llm_nature.journey --out "$OUT"
python -m llm_nature.verify --out "$OUT" > "$OUT/manifest.pretty.json"
echo "[OK] wrote $OUT/PRINTED_OUTPUT_ALL_STAGES.txt and $OUT/manifest.json"
