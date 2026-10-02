"""ID-switch failure analysis: classify every ID switch by cause and render before/after strips."""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from ..algorithm.geometry import ioa_matrix
from ..datasets.mot_io import xywh_to_xyxy
from ..inference.cache import DetSet, gt_eval_rows
from ..utils import ensure_dir, get_logger, resolve_path
from ..evaluation.metrics import evaluate
from .draw import crop_strip

log = get_logger()
CAUSES = ("swap_with_other_id", "overlapped", "after_gap", "other")


def classify_switch(gt: np.ndarray, preds: np.ndarray, frame: int, gid: int, prev_id: int, new_id: int) -> str:
    """Causes (first match wins):
    swap_with_other_id : the new id belonged to a different GT person within the last 30 frames (identity drift);
    overlapped         : the GT box is >50% covered by another GT box at the switch frame;
    after_gap          : the identity was unmatched for >=5 frames before the switch (re-association);
    other.
    """
    g = gt[(gt[:, 6] != 0) & (gt[:, 7] == 1)]
    gf = g[g[:, 0] == frame]
    me = gf[gf[:, 1] == gid]
    if len(preds):
        past = preds[(preds[:, 0] >= frame - 30) & (preds[:, 0] < frame) & (preds[:, 1] == new_id)]
        for r in past:
            gr = g[g[:, 0] == r[0]]
            if len(gr):
                from ..algorithm.geometry import iou_matrix
                iou = iou_matrix(xywh_to_xyxy(r[None, 2:6]), xywh_to_xyxy(gr[:, 2:6]))[0]
                j = int(np.argmax(iou))
                if iou[j] >= 0.5 and int(gr[j, 1]) != gid:
                    return "swap_with_other_id"
    if len(me) and len(gf) > 1:
        b = xywh_to_xyxy(gf[:, 2:6])
        m = ioa_matrix(b, b)
        np.fill_diagonal(m, 0)
        if m[int(np.where(gf[:, 1] == gid)[0][0])].max() > 0.5:
            return "overlapped"
    if len(preds):
        prev = preds[(preds[:, 0] >= frame - 5) & (preds[:, 0] < frame) & (preds[:, 1] == prev_id)]
        if len(prev) == 0:
            return "after_gap"
    return "other"


def failure_report(detsets: List[DetSet], preds_by_method: Dict[str, Dict[str, np.ndarray]], out_dir, thr: float = 0.5,
                   n_strips: int = 4, reference: str = "b1", target: str = "rca") -> pd.DataFrame:
    out = ensure_dir(out_dir)
    rows, strip_jobs = [], []
    for ds in detsets:
        gt = gt_eval_rows(ds.spec)
        for m, preds in preds_by_method.items():
            r = evaluate(gt, preds[ds.spec.name], (ds.spec.frame_start, ds.spec.frame_end), thr, return_events=True)
            for (f, gid, p, n) in r["events"]:
                cause = classify_switch(gt, preds[ds.spec.name], f, gid, p, n)
                rows.append({"seq": ds.spec.name, "method": m, "frame": f, "gt_id": gid, "prev_pred": p, "new_pred": n, "cause": cause})
    df = pd.DataFrame(rows, columns=["seq", "method", "frame", "gt_id", "prev_pred", "new_pred", "cause"])
    df.to_csv(out / "id_switches.csv", index=False)
    summary = (df.groupby(["method", "cause"]).size().unstack(fill_value=0).reindex(columns=list(CAUSES), fill_value=0)
               if len(df) else pd.DataFrame(columns=list(CAUSES)))
    summary.to_csv(out / "idsw_by_cause.csv")
    # strips: switches of the reference method (and what the target method does at the same moment)
    if reference in preds_by_method and len(df):
        by_seq = {ds.spec.name: ds for ds in detsets}
        ref_sw = df[df.method == reference].head(n_strips)
        for k, e in enumerate(ref_sw.itertuples()):
            ds = by_seq[e.seq]
            gt = gt_eval_rows(ds.spec)
            g = gt[(gt[:, 0] == e.frame) & (gt[:, 1] == e.gt_id)]
            if not len(g):
                continue
            box = xywh_to_xyxy(g[0, 2:6])
            frames = [max(ds.spec.frame_start, e.frame - 10), max(ds.spec.frame_start, e.frame - 1),
                      e.frame, min(ds.spec.frame_end, e.frame + 10)]
            sel = {m: preds_by_method[m][e.seq] for m in (reference, target) if m in preds_by_method}
            img = crop_strip(ds.spec, sel, box, frames)
            cv2.imwrite(str(out / f"switch_{k:02d}_{e.seq}_f{e.frame}_{e.cause}.png"), img)
    return summary
