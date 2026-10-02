"""Dataset registry: locate sequences and define the train/val splits used by every experiment.

mot17 / mot20 : official *train* sequences; split = first half (train) / second half (val) of each sequence
                (the ByteTrack-style half-split protocol). MOTChallenge test sets have no public GT.
dancetrack    : official train / val directories.
synthetic     : generated demo data, train / val directories.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import resolve_path
from .mot_io import read_seqinfo


class DatasetMissingError(RuntimeError):
    pass


@dataclass
class SequenceSpec:
    dataset: str
    name: str
    dir: Path
    frame_start: int
    frame_end: int
    width: int
    height: int
    fps: int
    img_dir: Path
    ext: str

    @property
    def gt_path(self) -> Path:
        return self.dir / "gt" / "gt.txt"

    @property
    def det_path(self) -> Path:
        return self.dir / "det" / "det.txt"

    @property
    def frames(self) -> range:
        return range(self.frame_start, self.frame_end + 1)

    def image_path(self, frame: int) -> Path:
        return self.img_dir / f"{frame:06d}{self.ext}"


MOT_SEQS = {
    "mot17": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"],
    "mot20": ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"],
}
HELP = {
    "mot17": "python scripts/download_mot17.py   (or download MOT17.zip from https://motchallenge.net/data/MOT17/ and unzip into data/MOT17)",
    "mot20": "python scripts/download_mot20.py   (or download MOT20.zip from https://motchallenge.net/data/MOT20/ and unzip into data/MOT20)",
    "dancetrack": "python scripts/download_dancetrack.py  (prints the manual steps; data lives on the official DanceTrack page)",
    "synthetic": "python main.py prepare --dataset synthetic",
}


def _spec(dataset: str, name: str, d: Path, start: Optional[int], end: Optional[int]) -> SequenceSpec:
    info = read_seqinfo(d)
    length = info["seqLength"] or (end or 0)
    if not length:
        raise DatasetMissingError(f"{d} has no seqinfo.ini / seqLength; dataset looks incomplete.")
    return SequenceSpec(dataset, name, d, start or 1, end or length, info["imWidth"], info["imHeight"],
                        info["frameRate"], d / info["imDir"], info["imExt"])


def list_sequences(dataset: str, split: str, data_root="data", mot17_detector: str = "FRCNN",
                   only: Optional[List[str]] = None) -> List[SequenceSpec]:
    dataset = dataset.lower()
    root = resolve_path(data_root)
    specs: List[SequenceSpec] = []
    if dataset in ("mot17", "mot20"):
        base = root / dataset.upper() / "train"
        if not base.exists():
            raise DatasetMissingError(f"{base} not found. Get the data with:\n  {HELP[dataset]}")
        for s in MOT_SEQS[dataset]:
            name = f"{s}-{mot17_detector}" if dataset == "mot17" else s
            d = base / name
            if not d.exists():
                raise DatasetMissingError(f"Sequence folder missing: {d}. Re-run:\n  {HELP[dataset]}")
            info = read_seqinfo(d)
            half = info["seqLength"] // 2
            start, end = (1, half) if split == "train" else (half + 1, info["seqLength"])
            specs.append(_spec(dataset, name, d, start, end))
    elif dataset in ("dancetrack", "synthetic"):
        sub = "train" if split == "train" else "val"
        base = root / ("dancetrack" if dataset == "dancetrack" else "synthetic") / sub
        if not base.exists():
            raise DatasetMissingError(f"{base} not found. Get the data with:\n  {HELP[dataset]}")
        for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "gt").exists()):
            specs.append(_spec(dataset, d.name, d, None, None))
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from mot17, mot20, dancetrack, synthetic.")
    if only:
        specs = [s for s in specs if s.name in only]
    if not specs:
        raise DatasetMissingError(f"No sequences found for {dataset}/{split} under {root}. Try: {HELP[dataset]}")
    return specs
