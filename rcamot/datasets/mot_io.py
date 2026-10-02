"""MOTChallenge-format readers/writers.

gt.txt : frame,id,x,y,w,h,flag,class,visibility
det.txt: frame,-1,x,y,w,h,score,-1,-1,-1
result : frame,id,x,y,w,h,score,-1,-1,-1
"""
from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict

import numpy as np

from ..utils import resolve_path


def read_mot_txt(path) -> np.ndarray:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"MOT file not found: {p}")
    if p.stat().st_size == 0:
        return np.zeros((0, 10))
    arr = np.loadtxt(p, delimiter=",", ndmin=2)
    return arr


def load_gt(path) -> np.ndarray:
    """-> (N, 9): frame, id, x, y, w, h, flag, class, visibility (missing columns filled with 1)."""
    a = read_mot_txt(path)
    out = np.ones((len(a), 9))
    out[:, : min(9, a.shape[1])] = a[:, :9]
    return out


def load_det(path) -> np.ndarray:
    """-> (N, 7): frame, id, x, y, w, h, score."""
    a = read_mot_txt(path)
    out = np.zeros((len(a), 7))
    if len(a):
        out[:, : min(7, a.shape[1])] = a[:, :7]
    return out


def write_mot_txt(path, rows: np.ndarray) -> Path:
    """rows: (N, >=7) [frame, id, x, y, w, h, score]."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows).reshape(-1, rows.shape[-1] if len(rows) else 7)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1\n" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    return p


def read_seqinfo(seq_dir) -> Dict:
    p = resolve_path(seq_dir) / "seqinfo.ini"
    info = {"imDir": "img1", "frameRate": 30, "seqLength": 0, "imWidth": 0, "imHeight": 0, "imExt": ".jpg"}
    if p.exists():
        cp = configparser.ConfigParser()
        cp.read(p)
        s = cp["Sequence"]
        info.update({"imDir": s.get("imDir", "img1"), "frameRate": int(float(s.get("frameRate", 30))),
                     "seqLength": int(s.get("seqLength", 0)), "imWidth": int(s.get("imWidth", 0)),
                     "imHeight": int(s.get("imHeight", 0)), "imExt": s.get("imExt", ".jpg")})
    return info


def write_seqinfo(seq_dir, name: str, length: int, fps: int, w: int, h: int, ext: str = ".jpg") -> None:
    p = resolve_path(seq_dir) / "seqinfo.ini"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"[Sequence]\nname={name}\nimDir=img1\nframeRate={fps}\nseqLength={length}\n"
                 f"imWidth={w}\nimHeight={h}\nimExt={ext}\n")


def xywh_to_xyxy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], a[..., 1], a[..., 0] + a[..., 2], a[..., 1] + a[..., 3]], axis=-1)


def xyxy_to_xywh(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.stack([a[..., 0], a[..., 1], a[..., 2] - a[..., 0], a[..., 3] - a[..., 1]], axis=-1)
