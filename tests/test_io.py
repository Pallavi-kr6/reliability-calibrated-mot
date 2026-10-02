import numpy as np
import pytest

from rcamot.datasets.mot_io import load_gt, read_mot_txt, read_seqinfo, write_mot_txt, write_seqinfo, xywh_to_xyxy, xyxy_to_xywh
from rcamot.datasets.registry import DatasetMissingError, list_sequences
from rcamot.utils import deep_update, load_config, parse_overrides, resolve_path, ROOT


def test_mot_roundtrip(tmp_path):
    rows = np.array([[1, 3, 10.5, 20.25, 30.0, 60.0, 0.9], [2, 3, 11.0, 21.0, 30.0, 60.0, 0.8]])
    p = write_mot_txt(tmp_path / "r.txt", rows)
    back = read_mot_txt(p)
    assert back.shape == (2, 10) and np.allclose(back[:, :7], rows, atol=0.01)


def test_gt_loader_fills_missing_columns(tmp_path):
    (tmp_path / "gt.txt").write_text("1,1,10,10,20,40\n")
    g = load_gt(tmp_path / "gt.txt")
    assert g.shape == (1, 9) and g[0, 7] == 1 and g[0, 8] == 1


def test_box_conversions():
    a = np.array([[10, 20, 30, 60.0]])
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(xyxy_to_xywh(a))), xyxy_to_xywh(a))


def test_seqinfo_roundtrip(tmp_path):
    write_seqinfo(tmp_path, "s", 100, 25, 640, 360)
    i = read_seqinfo(tmp_path)
    assert i["seqLength"] == 100 and i["imWidth"] == 640 and i["frameRate"] == 25


def test_missing_dataset_error_is_actionable(tmp_path):
    with pytest.raises(DatasetMissingError, match="download_mot17"):
        list_sequences("mot17", "val", tmp_path)


def test_paths_resolve_against_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("data/x") == ROOT / "data" / "x"
    assert resolve_path(str(tmp_path / "abs")) == tmp_path / "abs"


def test_config_merge_and_overrides():
    c = load_config("configs/rca.yaml", dataset="synthetic", overrides=parse_overrides(["tracker.max_age=77", "rca.use_occ=false"]))
    assert c["method"] == "rca" and c["tracker"]["max_age"] == 77 and c["rca"]["use_occ"] is False
    assert c["results_dir"] == "results/demo" and c["data"]["embedder"] == "colorhist"
    assert c["tracker"]["ema_alpha"] == 0.9      # inherited from default.yaml


def test_deep_update_does_not_mutate():
    a = {"x": {"y": 1}}
    b = deep_update(a, {"x": {"z": 2}})
    assert a == {"x": {"y": 1}} and b == {"x": {"y": 1, "z": 2}}


def test_no_absolute_personal_paths_in_configs():
    for p in (ROOT / "configs").rglob("*.yaml"):
        t = p.read_text()
        assert "C:\\" not in t and "/home/" not in t and "/Users/" not in t, p
