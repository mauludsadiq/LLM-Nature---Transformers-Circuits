# LLM Nature — Transformers + Circuits
**Complete Transformer Interpretability Journey** — from first principles to GPT‑2 “brain surgery”.

This repo is built so a human can read the **PRINTED OUTPUT** and see the staged development, while the machine can verify the produced artifacts.

Core premise:
> LLMs function like the friend who attempts to finish your sentences — often confidently, and sometimes wrong.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the full journey (prints stage outputs + saves figures)
python -m llm_nature.journey --out _out

# Verify outputs (recomputes file hashes)
python -m llm_nature.verify --out _out
```

## Human-facing artifact
After running the journey, read:

- `_out/PRINTED_OUTPUT_ALL_STAGES.txt`

Each stage prints:
- Banner title
- “Friend finishes your sentence” line (shows progression)
- Key numerical/structural outputs for that stage

Stage 7 runs real GPT‑2 weights and prints top‑10 next-token probabilities **before vs after** activation patching.

## Machine-facing artifacts
Each stage writes:
- `printed_output.txt` (same text that was printed)
- `artifacts.json` (structured values)
Figures are saved as `.png`.

Verification writes `_out/manifest.json` with SHA256 digests.
