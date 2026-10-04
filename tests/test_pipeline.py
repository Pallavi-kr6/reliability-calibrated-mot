"""Scripted scene (two similar people cross with overlap, one disappears and reappears) + rendered-data integration."""
import numpy as np
import pytest

from rcamot.evaluation.metrics import evaluate
from tests.helpers import GAP, N_FRAMES, run_scene


def ids_of(rows, gt_id_by_x=None):
    return rows


def person_ids(rows, gt):
    """For every GT identity, the set of predicted ids matched to it."""
    from rcamot.algorithm.geometry import iou_matrix
    out = {1: [], 2: []}
    for t in range(1, N_FRAMES + 1):
        g, p = gt[gt[:, 0] == t], rows[rows[:, 0] == t]
        if len(g) and len(p):
            iou = iou_matrix(np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]],
                             np.c_[p[:, 2], p[:, 3], p[:, 2] + p[:, 4], p[:, 3] + p[:, 5]])
            for a in range(len(g)):
                if iou[a].max() >= 0.5:
                    out[int(g[a, 1])].append((t, int(p[iou[a].argmax(), 1])))
    return out


@pytest.mark.parametrize("method", ["b0", "b1", "b2", "b3"])
def test_crossing_similar_people_keep_identities(method):
    rows, gt = run_scene(method)
    r = evaluate(gt, rows, (1, N_FRAMES))
    assert r["IDSW"] <= 1 and r["MOTA"] > 90, (method, r["IDSW"], r["MOTA"])


@pytest.mark.parametrize("method", ["b0", "b1", "b3"])
def test_disappearing_person_regains_original_id(method):
    rows, gt = run_scene(method)
    ids = person_ids(rows, gt)
    before = {i for t, i in ids[1] if t < GAP[0]}
    after = {i for t, i in ids[1] if t >= GAP[1]}
    assert len(before) == 1 and before == after, (method, before, after)


def test_rca_on_scripted_scene_with_fitted_model(tmp_path):
    """Fit RCA on pairs mined from the scripted scene itself and check the tracker behaves (smoke + sanity)."""
    from rcamot.algorithm.features import FEATURE_KEYS
    from rcamot.models.rca import RCAModel, fit_platt, fit_rca
    rows0, gt = run_scene("b1")
    # build labelled pairs from consecutive frames of the scripted scene
    from rcamot.algorithm.features import compute_pair_features
    from tests.helpers import scripted_scene
    frames, _ = scripted_scene()
    X = {k: [] for k in FEATURE_KEYS}; y = []
    G = {"iou": 0.0, "ctr": 0.75, "growth": 0.1, "ctr_max": 2.5}
    for (t, b, s, e, ids), (t2, b2, s2, e2, ids2) in zip(frames[:-1], frames[1:]):
        f = compute_pair_features(b, np.ones(len(b)), e, b2, s2, e2, np.zeros(len(b2)), np.zeros(len(b)), G)
        for k, v in f.as_dict().items():
            X[k].append(v[f.mask])
        y.append((ids[:, None] == ids2[None, :])[f.mask].astype(float))
    X = {k: np.concatenate(v) for k, v in X.items()}
    y = np.concatenate(y)
    th, _ = fit_rca(X, y)
    m = RCAModel(th)
    m.platt_a, m.platt_b = fit_platt(m.logit(X), y)
    rows, gt = run_scene("rca", {"rca": {"accept_thresh": 0.3}}, m)
    r = evaluate(gt, rows, (1, N_FRAMES))
    assert r["IDSW"] <= 2 and r["MOTA"] > 80, r


def test_end_to_end_rendered_data_all_methods(tmp_path):
    """Render tiny synthetic sequences -> cache -> train RCA -> track B0..B3/RCA -> metrics exist and are sane."""
    from rcamot.datasets.synthetic import crowd_sequence, story_sequence
    from rcamot.evaluation.runner import run_experiment, method_cfg
    from rcamot.inference.pipeline import get_detsets
    from rcamot.training.train_rca import train_rca
    root = tmp_path / "data"
    crowd_sequence(root / "synthetic" / "train" / "t0", "t0", 5, 60, 0.5, 1, W=320, H=240)
    crowd_sequence(root / "synthetic" / "train" / "t1", "t1", 8, 60, 0.2, 2, W=320, H=240)
    story_sequence(root / "synthetic" / "val" / "story", "story", 60, W=320, H=240)
    ov = {"data": {"root": str(root)}, "cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "res"),
          "rca": {"model_path": str(tmp_path / "rca.json"), "fit_block": 20}, "_no_tuned": True}
    model, _ = train_rca(method_cfg("rca", "synthetic", ov))
    assert model.meta["n_pairs"] > 100
    for m in ("b0", "b1", "b2", "b3", "rca"):
        cfg = method_cfg(m, "synthetic", ov)
        res = run_experiment(cfg, "val", get_detsets(cfg, "val"), model if m == "rca" else None)
        c = res["combined"]
        assert c["num_gt"] > 0 and 0 <= c["HOTA"] <= 100 and np.isfinite(c["MOTA"]), (m, c)
        assert (tmp_path / "res" / "runs" / "synthetic" / m / "metrics.json").exists()


def test_cache_is_reused_and_invalidated(tmp_path):
    from rcamot.datasets.synthetic import crowd_sequence
    from rcamot.evaluation.runner import method_cfg
    from rcamot.inference.pipeline import get_detsets
    root = tmp_path / "data"
    crowd_sequence(root / "synthetic" / "val" / "v0", "v0", 4, 30, 0.5, 3, W=320, H=240)
    ov = {"data": {"root": str(root)}, "cache_dir": str(tmp_path / "cache"), "_no_tuned": True}
    cfg = method_cfg("b1", "synthetic", ov)
    get_detsets(cfg, "val")
    f = next((tmp_path / "cache").rglob("*.npz"))
    m1 = f.stat().st_mtime_ns
    get_detsets(cfg, "val")
    assert f.stat().st_mtime_ns == m1                                   # reused
    cfg2 = method_cfg("b1", "synthetic", {**ov, "data": {"root": str(root), "det_min_score": 0.3}})
    get_detsets(cfg2, "val")
    assert f.stat().st_mtime_ns != m1                                   # fingerprint changed -> recomputed


def test_mot17_shaped_tree_half_split_and_tracking(tmp_path):
    """Build a MOT17-shaped folder (7 sequences) from tiny rendered scenes and run the real-dataset code path:
    registry half-split (first half = train, second half = val), public det.txt, cache, tracking, evaluation."""
    from rcamot.datasets.registry import MOT_SEQS, list_sequences
    from rcamot.datasets.synthetic import crowd_sequence
    from rcamot.evaluation.runner import method_cfg, run_experiment
    from rcamot.inference.pipeline import get_detsets
    root = tmp_path / "data"
    for k, s in enumerate(MOT_SEQS["mot17"]):
        name = f"{s}-FRCNN"
        crowd_sequence(root / "MOT17" / "train" / name, name, 4, 24, 0.5, k, W=320, H=240)
    tr = list_sequences("mot17", "train", root)
    va = list_sequences("mot17", "val", root)
    assert len(tr) == len(va) == 7
    assert [s.name for s in list_sequences("mot17", "val", root, only=["MOT17-04-FRCNN"])] == ["MOT17-04-FRCNN"]
    assert (tr[0].frame_start, tr[0].frame_end) == (1, 12) and (va[0].frame_start, va[0].frame_end) == (13, 24)
    ov = {"data": {"root": str(root), "embedder": "colorhist"}, "cache_dir": str(tmp_path / "cache"),
          "results_dir": str(tmp_path / "res"), "_no_tuned": True}
    cfg = method_cfg("b1", "mot17", ov)
    res = run_experiment(cfg, "val", get_detsets(cfg, "val", only=["MOT17-02-FRCNN", "MOT17-04-FRCNN"]))
    assert set(res["per_sequence"]) == {"MOT17-02-FRCNN", "MOT17-04-FRCNN"} and res["combined"]["num_gt"] > 0
