"""Official TrackEval evaluation (https://github.com/JonathonLuiten/TrackEval, MIT licence).

TrackEval is NOT bundled. Clone it with `python scripts/setup_trackeval.py` (-> third_party/TrackEval) or set the
environment variable TRACKEVAL_DIR. Half-sequence splits are exported with frame numbers shifted to start at 1,
which is what TrackEval expects for a sequence of the given length.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..datasets.mot_io import load_gt, write_mot_txt, write_seqinfo
from ..datasets.registry import SequenceSpec
from ..utils import ROOT, get_logger

log = get_logger()


def find_trackeval() -> Path:
    cands = [os.environ.get("TRACKEVAL_DIR"), str(ROOT / "third_party" / "TrackEval")]
    for c in cands:
        if c and (Path(c) / "trackeval").exists():
            return Path(c)
    raise FileNotFoundError("TrackEval not found. Run `python scripts/setup_trackeval.py` (clones it into third_party/TrackEval) "
                            "or set TRACKEVAL_DIR. Until then use `--backend own`.")


def evaluate_with_trackeval(specs: List[SequenceSpec], preds: Dict[str, np.ndarray], cfg: Dict) -> Dict:
    root = find_trackeval()
    sys.path.insert(0, str(root))
    for n, t in (("float", float), ("int", int), ("bool", bool)):  # TrackEval predates NumPy>=1.24
        if not hasattr(np, n):
            setattr(np, n, t)
    import trackeval  # type: ignore

    tmp = Path(tempfile.mkdtemp(prefix="rcamot_te_"))
    try:
        gt_dir, tr_dir = tmp / "gt", tmp / "trackers" / "rcamot"
        gt_dir.mkdir(parents=True)
        tr_dir.mkdir(parents=True)
        for s in specs:
            shift = s.frame_start - 1
            g = load_gt(s.gt_path)
            g = g[(g[:, 0] >= s.frame_start) & (g[:, 0] <= s.frame_end)].copy()
            g[:, 0] -= shift
            sd = gt_dir / s.name
            (sd / "gt").mkdir(parents=True)
            np.savetxt(sd / "gt" / "gt.txt", g, delimiter=",", fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%d,%d,%.4f")
            write_seqinfo(sd, s.name, s.frame_end - s.frame_start + 1, s.fps, s.width, s.height)
            p = preds[s.name].copy()
            if len(p):
                p[:, 0] -= shift
            write_mot_txt(tr_dir / f"{s.name}.txt", p if len(p) else np.zeros((0, 7)))
        (tmp / "seqmap.txt").write_text("name\n" + "\n".join(s.name for s in specs) + "\n")
        ec = trackeval.Evaluator.get_default_eval_config()
        ec.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "PRINT_ONLY_COMBINED": True,
                   "TIME_PROGRESS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False})
        dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dc.update({"GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(tmp / "trackers"), "TRACKERS_TO_EVAL": ["rcamot"],
                   "TRACKER_SUB_FOLDER": "", "OUTPUT_SUB_FOLDER": "", "SKIP_SPLIT_FOL": True, "SPLIT_TO_EVAL": "train",
                   "BENCHMARK": "MOT17", "SEQMAP_FILE": str(tmp / "seqmap.txt"), "PRINT_CONFIG": False, "DO_PREPROC": True})
        ev = trackeval.Evaluator(ec)
        metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False}), trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
                   trackeval.metrics.Identity({"PRINT_CONFIG": False})]
        res, _ = ev.evaluate([trackeval.datasets.MotChallenge2DBox(dc)], metrics)
        r = res["MotChallenge2DBox"]["rcamot"]

        def pick(x: Dict) -> Dict:
            return {"HOTA": float(np.mean(x["HOTA"]["HOTA"]) * 100), "DetA": float(np.mean(x["HOTA"]["DetA"]) * 100),
                    "AssA": float(np.mean(x["HOTA"]["AssA"]) * 100), "IDF1": float(x["Identity"]["IDF1"] * 100),
                    "MOTA": float(x["CLEAR"]["MOTA"] * 100), "IDSW": int(x["CLEAR"]["IDSW"])}

        out = {"backend": "trackeval", "combined": pick(r["COMBINED_SEQ"]["pedestrian"]),
               "per_sequence": {s.name: pick(r[s.name]["pedestrian"]) for s in specs}}
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
