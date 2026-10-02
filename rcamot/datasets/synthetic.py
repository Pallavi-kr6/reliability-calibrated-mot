"""Synthetic MOT sequences rendered as real image frames (labelled DEMO data, never benchmark data).

People are drawn as head + torso + legs silhouettes with depth ordering and a static wall occluder, so
(a) occlusion contaminates appearance crops physically, (b) similar-clothing identities exist,
(c) people disappear and reappear. Output uses the MOTChallenge layout so the whole pipeline is identical to
the real-dataset path:  <root>/<split>/<seq>/{img1/*.jpg, gt/gt.txt, det/det.txt, seqinfo.ini}.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import get_logger, resolve_path
from .mot_io import write_seqinfo

log = get_logger()
OCC_ID = 65535


@dataclass
class SynPerson:
    pid: int
    shirt: Tuple[int, int, int]
    pants: Tuple[int, int, int]
    xs: np.ndarray
    ys: np.ndarray                      # foot position per frame
    present: np.ndarray                 # False while the person is away (disappearance)


def _clip255(c):
    return tuple(int(np.clip(v, 0, 255)) for v in c)


def _colors(rng: np.random.Generator, n: int, spread: float):
    """Shirt/pants colours (BGR). `spread` in [0,1]: 0 = everybody in one uniform, 1 = highly diverse."""
    base = rng.integers(40, 215, 3).astype(float)
    shirts, pants = [], []
    for _ in range(n):
        shirts.append(_clip255(base * (1 - spread) + rng.integers(20, 235, 3) * spread + rng.normal(0, 6, 3)))
        pants.append(_clip255(np.array([50, 50, 60]) * (1 - spread) + rng.integers(15, 120, 3) * spread + rng.normal(0, 4, 3)))
    return shirts, pants


def _walk(rng, n_frames, W, H, speed=2.2):
    x, y = rng.uniform(40, W - 40), rng.uniform(150, H - 15)
    vx, vy = rng.normal(0, speed), rng.normal(0, speed * 0.3)
    xs, ys = np.zeros(n_frames), np.zeros(n_frames)
    for t in range(n_frames):
        vx = 0.96 * vx + rng.normal(0, 0.25)
        vy = 0.96 * vy + rng.normal(0, 0.08)
        x, y = x + vx, y + vy
        if x < 30 or x > W - 30:
            vx, x = -vx, np.clip(x, 30, W - 30)
        if y < 150 or y > H - 8:
            vy, y = -vy, np.clip(y, 150, H - 8)
        xs[t], ys[t] = x, y
    return xs, ys


def _height(y: float) -> float:
    return 50.0 + (y - 140.0) * 0.5


def _draw_person(canvas, idmap, cx, foot, h, shirt, pants, pid):
    w = 0.19 * h
    torso = (int(cx - w), int(foot - 0.9 * h)), (int(cx + w), int(foot - 0.5 * h))
    legs = (int(cx - 0.15 * h), int(foot - 0.5 * h)), (int(cx + 0.15 * h), int(foot))
    head = (int(cx), int(foot - 0.93 * h)), max(2, int(0.08 * h))
    for img, cols in ((canvas, (pants, shirt, (150, 170, 210))), (idmap, (pid + 1, pid + 1, pid + 1))):
        if img is None:
            continue
        cv2.rectangle(img, legs[0], legs[1], cols[0], -1)
        cv2.rectangle(img, torso[0], torso[1], cols[1], -1)
        cv2.circle(img, head[0], head[1], cols[2], -1)
    return (cx - w, foot - h, cx + w, foot)


def _full_area(h: float) -> float:
    m = np.zeros((int(h) + 20, int(0.6 * h) + 20), np.uint8)
    _draw_person(None, m, m.shape[1] / 2, h + 5, h, 0, 0, 0)
    return float((m > 0).sum()) + 1e-9


def render_sequence(out_dir, name: str, persons: List[SynPerson], n_frames: int, W: int, H: int,
                    occluders: List[Tuple[int, int, int, int]], seed: int, fps: int = 25) -> Dict:
    """Render frames + gt + noisy 'detector' output for the given people."""
    rng = np.random.default_rng(seed + 1000)
    out = resolve_path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "img1").mkdir(parents=True)
    (out / "gt").mkdir()
    (out / "det").mkdir()
    yy = np.linspace(170, 120, H)[:, None, None] * np.ones((H, W, 3))
    bg = np.clip(yy + np.array([5, 0, -5]), 0, 255).astype(np.uint8)
    gt_rows, det_rows = [], []
    for t in range(n_frames):
        canvas = bg.copy()
        idmap = np.zeros((H, W), np.uint16)
        order = sorted([p for p in persons if p.present[t]], key=lambda p: p.ys[t])
        boxes = {}
        for p in order:
            h = _height(p.ys[t])
            boxes[p.pid] = (_draw_person(canvas, idmap, p.xs[t], p.ys[t], h, p.shirt, p.pants, p.pid), h)
        for (x1, y1, x2, y2) in occluders:
            canvas[y1:y2, x1:x2] = (95, 100, 105)
            idmap[y1:y2, x1:x2] = OCC_ID
        canvas = np.clip(canvas.astype(np.float32) + rng.normal(0, 3.0, canvas.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "img1" / f"{t + 1:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for p in order:
            (x1, y1, x2, y2), h = boxes[p.pid]
            vis = float((idmap == p.pid + 1).sum()) / _full_area(h)
            vis = float(np.clip(vis, 0, 1))
            gt_rows.append([t + 1, p.pid + 1, x1, y1, x2 - x1, y2 - y1, 1, 1, vis])
            # --- noisy detector: box shrinks to the visible part, misses/scores depend on visibility ---
            if vis < 0.12 or rng.random() < 0.02 + 0.6 * (1 - vis) ** 2:
                continue
            ys_, xs_ = np.where(idmap == p.pid + 1)
            vx1, vy1, vx2, vy2 = (xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1) if vis < 0.8 else (x1, y1, x2, y2)
            bw, bh = vx2 - vx1, vy2 - vy1
            j = rng.normal(0, 0.04, 4) * np.array([bw, bh, bw, bh])
            score = float(np.clip(0.35 + 0.65 * vis ** 0.7 + rng.normal(0, 0.04), 0.12, 0.99))
            det_rows.append([t + 1, -1, vx1 + j[0], vy1 + j[1], bw + j[2] - j[0], bh + j[3] - j[1], score, -1, -1, -1])
        if rng.random() < 0.25:  # occasional false positive
            fx, fy = rng.uniform(20, W - 60), rng.uniform(100, H - 90)
            det_rows.append([t + 1, -1, fx, fy, rng.uniform(25, 45), rng.uniform(60, 90), rng.uniform(0.12, 0.45), -1, -1, -1])
    np.savetxt(out / "gt" / "gt.txt", np.array(gt_rows), delimiter=",", fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.4f")
    np.savetxt(out / "det" / "det.txt", np.array(det_rows), delimiter=",",
               fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d")
    write_seqinfo(out, name, n_frames, fps, W, H)
    return {"name": name, "frames": n_frames, "people": len(persons)}


def crowd_sequence(out_dir, name: str, n_people: int, n_frames: int, spread: float, seed: int,
                   W: int = 640, H: int = 360, gap_prob: float = 0.5, wall: bool = True) -> Dict:
    rng = np.random.default_rng(seed)
    shirts, pants = _colors(rng, n_people, spread)
    persons = []
    for i in range(n_people):
        xs, ys = _walk(rng, n_frames, W, H)
        present = np.ones(n_frames, bool)
        if rng.random() < gap_prob and n_frames > 60:  # leaves the scene for a while and returns (same identity)
            k = int(rng.integers(8, 35))
            t0 = int(rng.integers(15, n_frames - k - 15))
            present[t0:t0 + k] = False
        persons.append(SynPerson(i, shirts[i], pants[i], xs, ys, present))
    occl = [(420, 110, 470, H)] if wall else []
    return render_sequence(out_dir, name, persons, n_frames, W, H, occl, seed)


def story_sequence(out_dir, name: str = "syn_story", n_frames: int = 150, seed: int = 7,
                   W: int = 640, H: int = 360) -> Dict:
    """Hand-scripted scene: two similar blue-shirt people CROSS (near-total overlap), one red bystander, one
    person walks behind the wall, and person A disappears for 15 frames then reappears."""
    t = np.arange(n_frames)
    blue_a, blue_b = (190, 90, 40), (180, 96, 46)          # BGR, nearly identical
    a = SynPerson(0, blue_a, (50, 50, 60), 60 + 4.0 * t, np.full(n_frames, 250.0) + 0.1 * t, np.ones(n_frames, bool))
    b = SynPerson(1, blue_b, (55, 50, 60), 600 - 4.2 * t, np.full(n_frames, 252.0) - 0.1 * t, np.ones(n_frames, bool))
    c = SynPerson(2, (40, 40, 200), (50, 60, 50), np.full(n_frames, 120.0), 200 + 1.0 * t, np.ones(n_frames, bool))
    d = SynPerson(3, (60, 170, 70), (40, 40, 40), 330 + 3.0 * t, np.full(n_frames, 300.0), np.ones(n_frames, bool))
    a.present[95:110] = False
    a.xs = np.where(a.present, a.xs, a.xs)
    return render_sequence(out_dir, name, [a, b, c, d], n_frames, W, H, [(420, 110, 470, H)], seed)


def generate_demo_dataset(root="data/synthetic", quick: bool = False, seed: int = 0) -> List[Dict]:
    """Train (calibration) sequences with diverse appearance regimes + val sequences with growing crowd density."""
    root = resolve_path(root)
    nf = 70 if quick else 160
    info = []
    # (name, n_people, spread)  -- spread controls how distinguishable the clothing is
    train = [("syn_train_a", 8, 0.7), ("syn_train_b", 12, 0.35), ("syn_train_c", 16, 0.15), ("syn_train_d", 20, 0.08)]
    val = [("syn_val_06_distinct", 6, 0.7), ("syn_val_12_mixed", 12, 0.3),
           ("syn_val_18_similar", 18, 0.12), ("syn_val_24_uniform", 24, 0.06)]
    if quick:
        train, val = train[:3], val[:3]
    for k, (n, p, s) in enumerate(train):
        log.info(f"[synthetic] rendering train/{n} ({p} people, {nf} frames)")
        info.append(crowd_sequence(root / "train" / n, n, p, nf, s, seed + 10 + k))
    for k, (n, p, s) in enumerate(val):
        log.info(f"[synthetic] rendering val/{n} ({p} people, {nf} frames)")
        info.append(crowd_sequence(root / "val" / n, n, p, nf, s, seed + 100 + k))
    log.info("[synthetic] rendering val/syn_story (scripted crossing / occlusion / disappearance)")
    info.append(story_sequence(root / "val" / "syn_story", n_frames=100 if quick else 150))
    return info
