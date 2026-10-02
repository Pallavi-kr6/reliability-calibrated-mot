"""Overlay rendering: side-by-side videos / GIFs, ID-switch strips."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..datasets.registry import SequenceSpec
from ..utils import get_logger, resolve_path

log = get_logger()


def id_color(i: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(i) * 9973 + 7)
    c = rng.integers(60, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_tracks(img: np.ndarray, rows: np.ndarray, title: str = "", thickness: int = 2) -> np.ndarray:
    out = img.copy()
    for r in rows:
        x, y, w, h = r[2:6]
        c = id_color(int(r[1]))
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), c, thickness)
        cv2.putText(out, str(int(r[1])), (int(x) + 2, int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(int(r[1])), (int(x) + 2, int(y) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    if title:
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _rows_at(preds: np.ndarray, f: int) -> np.ndarray:
    return preds[preds[:, 0] == f] if len(preds) else np.zeros((0, 7))


def render_comparison(spec: SequenceSpec, preds_by_method: Dict[str, np.ndarray], out_prefix,
                      frames: Optional[range] = None, scale: float = 1.0, fps: int = 10, gif: bool = True) -> Dict[str, str]:
    """Side-by-side overlay of up to 3 methods. Writes <out_prefix>.mp4 (if the codec exists) and a GIF."""
    frames = frames or spec.frames
    names = list(preds_by_method)
    prefix = resolve_path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tiles_all = []
    for f in frames:
        img = cv2.imread(str(spec.image_path(f)))
        if img is None:
            continue
        tiles = [draw_tracks(img, _rows_at(preds_by_method[n], f), f"{n}   frame {f}") for n in names]
        tile = np.hstack(tiles)
        if scale != 1.0:
            tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tiles_all.append(tile)
    out = {}
    if not tiles_all:
        raise FileNotFoundError(f"No frames could be read for {spec.name}")
    h, w = tiles_all[0].shape[:2]
    mp4 = str(prefix) + ".mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if vw.isOpened():
        for t in tiles_all:
            vw.write(t)
        vw.release()
        out["mp4"] = mp4
    else:
        log.warning("OpenCV could not open an mp4 writer; only the GIF is produced.")
    if gif:
        step = max(1, len(tiles_all) // 45)
        ims = [Image.fromarray(cv2.cvtColor(cv2.resize(t, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
               .quantize(colors=48, method=Image.Quantize.MEDIANCUT) for t in tiles_all[::step]]
        gp = str(prefix) + ".gif"
        ims[0].save(gp, save_all=True, append_images=ims[1:], duration=int(1000 / fps) * step, loop=0, optimize=True)
        out["gif"] = gp
    return out


def crop_strip(spec: SequenceSpec, rows_by_method: Dict[str, np.ndarray], center_box: np.ndarray, frames: List[int],
               pad: float = 1.0, size=(120, 200)) -> np.ndarray:
    """One row per method, one column per frame, cropped around `center_box` (xyxy)."""
    x1, y1, x2, y2 = center_box
    w, h = x2 - x1, y2 - y1
    cx1, cy1, cx2, cy2 = int(x1 - pad * w), int(y1 - 0.3 * h), int(x2 + pad * w), int(y2 + 0.3 * h)
    lines = []
    for n, rows in rows_by_method.items():
        cells = []
        for f in frames:
            img = cv2.imread(str(spec.image_path(f)))
            if img is None:
                img = np.zeros((spec.height, spec.width, 3), np.uint8)
            img = draw_tracks(img, _rows_at(rows, f), "")
            H, W = img.shape[:2]
            c = img[max(cy1, 0):min(cy2, H), max(cx1, 0):min(cx2, W)]
            c = cv2.resize(c if c.size else np.zeros((10, 10, 3), np.uint8), (size[0] * 2, size[1]))
            cv2.putText(c, f"{n} f{f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(c)
        lines.append(np.hstack(cells))
    return np.vstack(lines)
