import os, tempfile
from llm_nature.util import RunContext
from llm_nature import stage1_attention, stage2_positional

def test_stage1_runs():
    with tempfile.TemporaryDirectory() as d:
        stage1_attention.run(RunContext(out_dir=d, seed=1))
        assert os.path.exists(os.path.join(d, "printed_output.txt"))
        assert os.path.exists(os.path.join(d, "artifacts.json"))

def test_stage2_writes_figure():
    with tempfile.TemporaryDirectory() as d:
        stage2_positional.run(RunContext(out_dir=d, seed=1))
        assert os.path.exists(os.path.join(d, "positional_encodings.png"))
