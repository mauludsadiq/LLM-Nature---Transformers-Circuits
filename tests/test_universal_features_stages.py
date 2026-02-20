from pathlib import Path

from llm_nature import stage11_universal_features as s11
from llm_nature import stage12_real_world_evidence as s12
from llm_nature import stage13_truth_geometry as s13
from llm_nature import stage14_universal_steering as s14
from llm_nature import stage15_complete_picture as s15


def _assert_file(p: Path) -> None:
    assert p.exists()
    assert p.is_file()
    assert p.stat().st_size > 0


def test_stage11_writes_expected_artifacts(tmp_path: Path) -> None:
    art = s11.run(tmp_path)
    d = tmp_path / "stage11_universal_features"
    _assert_file(d / "printed_output.txt")
    _assert_file(d / "artifacts.json")
    _assert_file(d / "similarity_matrix.png")
    sm = art["similarity_matrix"]
    assert len(sm) == 5
    assert len(sm[0]) == 5
    v = float(art["avg_cross_model_similarity_model0_model1"])
    assert 0.0 <= v <= 1.0
    txt = (d / "printed_output.txt").read_text(encoding="utf-8")
    assert "EXPERIMENT 11" in txt
    assert "You: The model is like a friend who tries to finish your sentence" in txt


def test_stage12_writes_expected_artifacts(tmp_path: Path) -> None:
    art = s12.run(tmp_path)
    d = tmp_path / "stage12_real_world_evidence"
    _assert_file(d / "printed_output.txt")
    _assert_file(d / "artifacts.json")
    _assert_file(d / "real_world_evidence.png")
    sm = art["similarity_matrix"]
    assert len(sm) == 3
    assert len(sm[0]) == 3


def test_stage13_writes_expected_artifacts(tmp_path: Path) -> None:
    art = s13.run(tmp_path)
    d = tmp_path / "stage13_truth_geometry"
    _assert_file(d / "printed_output.txt")
    _assert_file(d / "artifacts.json")
    _assert_file(d / "truth_projection_hist.png")
    sep = float(art["separation"])
    assert sep > 0.0


def test_stage14_writes_expected_artifacts(tmp_path: Path) -> None:
    art = s14.run(tmp_path)
    d = tmp_path / "stage14_universal_steering"
    _assert_file(d / "printed_output.txt")
    _assert_file(d / "artifacts.json")
    _assert_file(d / "transfer_effectiveness.png")
    m = float(art["mean_effectiveness"])
    assert 0.5 <= m <= 1.0


def test_stage15_writes_expected_artifacts(tmp_path: Path) -> None:
    art = s15.run(tmp_path)
    d = tmp_path / "stage15_complete_picture"
    _assert_file(d / "printed_output.txt")
    _assert_file(d / "artifacts.json")
    _assert_file(d / "complete_picture.png")
    assert art["stage"] == 15
