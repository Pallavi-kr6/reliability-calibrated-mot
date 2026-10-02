"""Download / extract / verify helpers for the public benchmark datasets (stdlib only)."""
from __future__ import annotations

import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from ..utils import get_logger, resolve_path

log = get_logger()

SOURCES = {
    "MOT17": {"url": "https://motchallenge.net/data/MOT17.zip", "page": "https://motchallenge.net/data/MOT17/",
              "license": "MOTChallenge data: research use, see the licence on the dataset page."},
    "MOT20": {"url": "https://motchallenge.net/data/MOT20.zip", "page": "https://motchallenge.net/data/MOT20/",
              "license": "MOTChallenge data: research use, see the licence on the dataset page."},
}


def _progress(done: int, total: int, t0: float) -> None:
    if total > 0:
        pct = 100 * done / total
        speed = done / max(time.time() - t0, 1e-6) / 1e6
        sys.stdout.write(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)  {speed:5.1f} MB/s")
        sys.stdout.flush()


def download(url: str, dest: Path, resume: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if (dest.exists() and resume) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "rcamot-downloader/0.1", **({"Range": f"bytes={have}-"} if have else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
            mode = "ab" if r.status == 206 else "wb"
            done = have if r.status == 206 else 0
            t0 = time.time()
            with open(dest, mode) as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    _progress(done, total, t0)
        print()
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return dest
        raise
    return dest


def extract_zip(zpath: Path, dest: Path, prefixes: Optional[Iterable[str]] = None) -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for name in names:
            if prefixes and not any(name.startswith(p) for p in prefixes):
                continue
            z.extract(name, dest)
            n += 1
    return n


def fetch_mot(name: str, root="data", include_test: bool = False, keep_zip: bool = False, zip_path: Optional[str] = None) -> Path:
    src = SOURCES[name]
    root = resolve_path(root)
    root.mkdir(parents=True, exist_ok=True)
    zp = Path(zip_path) if zip_path else root / f"{name}.zip"
    if not zp.exists():
        log.info(f"Downloading {name} from {src['url']} (several GB; resumable). {src['license']}")
        try:
            download(src["url"], zp)
        except Exception as e:
            raise RuntimeError(
                f"Automatic download failed ({type(e).__name__}: {e}).\nManual steps:\n"
                f"  1. open {src['page']} and download {name}.zip\n  2. run: python scripts/download_{name.lower()}.py --zip PATH/TO/{name}.zip\n"
                f"     (or unzip it so that {root / name / 'train'} exists)")
    if not zipfile.is_zipfile(zp):
        raise RuntimeError(f"{zp} is not a valid zip (interrupted download?). Delete it and retry.")
    log.info(f"Extracting {zp.name} -> {root} ({'train+test' if include_test else 'train only; test has no public GT'})")
    n = extract_zip(zp, root, None if include_test else [f"{name}/train/", f"{name}/seqmaps/"])
    log.info(f"Extracted {n} files")
    if not keep_zip and not zip_path:
        zp.unlink(missing_ok=True)
    return root / name


def normalise_dancetrack(zips: List[str], root="data") -> Path:
    """Extract manually downloaded DanceTrack zips and arrange them as data/dancetrack/{train,val}/<seq>/..."""
    root = resolve_path(root) / "dancetrack"
    raw = root / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for z in zips:
        zp = Path(z)
        if not zipfile.is_zipfile(zp):
            raise RuntimeError(f"{zp} is not a valid zip file")
        sub = raw / zp.stem
        log.info(f"Extracting {zp.name}")
        extract_zip(zp, sub)
    moved = 0
    for img1 in raw.rglob("img1"):
        seq = img1.parent
        hint = "/".join(p.lower() for p in seq.relative_to(raw).parts)
        split = "val" if "val" in hint else ("train" if "train" in hint else ("test" if "test" in hint else None))
        if split is None or split == "test":
            continue
        target = root / split / seq.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(seq), str(target))
            moved += 1
    shutil.rmtree(raw, ignore_errors=True)
    log.info(f"Arranged {moved} DanceTrack sequences under {root}")
    return root
