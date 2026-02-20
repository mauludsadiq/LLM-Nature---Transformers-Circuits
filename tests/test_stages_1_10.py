from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_nature.util import RunContext
from llm_nature import stage1_attention as s1
from llm_nature import stage2_positional as s2
from llm_nature import stage3_block as s3
from llm_nature import stage4_causal_mask as s4
from llm_nature import stage5_logit_lens as s5
from llm_nature import stage6_ioi_circuit as s6
from llm_nature import stage8_sae as s8
from llm_nature import stage9_vsa as s9
from llm_nature import stage10_pipeline as s10


@pytest.mark.parametrize(
    "mod,stage_num",
    [
        (s1, 1),
        (s2, 2),
        (s3, 3),
        (s4, 4),
        (s5, 5),
        (s6, 6),
        (s8, 8),
        (s9, 9),
        (s10, 10),
    ],
)
def test_stage_runs_and_writes_artifacts(tmp_path: Path, mod, stage_num: int) -> None:
    ctx = RunContext(out_dir=str(tmp_path), seed=42)
    r = mod.run(ctx)
    assert (tmp_path / "printed_output.txt").exists()
    assert (tmp_path / "artifacts.json").exists()
    assert int(r["stage"]) == stage_num


def test_stage7_optional(tmp_path: Path) -> None:
    run_stage7 = os.environ.get("LLM_NATURE_RUN_STAGE7", "") == "1"
    if not run_stage7:
        pytest.skip("set LLM_NATURE_RUN_STAGE7=1 to run GPT-2 patching test")

    from llm_nature import stage7_patching as s7

    ctx = RunContext(out_dir=str(tmp_path), seed=0)
    r = s7.run(ctx)
    assert int(r["stage"]) == 7
    assert (tmp_path / "printed_output.txt").exists()
    assert (tmp_path / "artifacts.json").exists()
