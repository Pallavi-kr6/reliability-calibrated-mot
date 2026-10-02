"""Shared helpers: paths, logging, seeding, config loading, hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent  # project root (works after extracting the ZIP anywhere)
CONFIG_DIR = ROOT / "configs"


def resolve_path(p: Any) -> Path:
    """Resolve a (possibly relative) path against the project root, never the CWD."""
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def ensure_dir(p: Any) -> Path:
    p = resolve_path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str = "rcamot", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(level)
        log.propagate = False
    return log


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def deep_update(base: Dict, new: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Any, dataset: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict:
    """Merge order: default.yaml -> (_base_ chain of `path`) -> datasets/<dataset>.yaml -> overrides."""
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    def with_base(p: Path) -> Dict:
        cfg = _load_yaml(p)
        base = cfg.pop("_base_", None)
        if base:
            return deep_update(with_base((p.parent / base).resolve()), cfg)
        return cfg

    cfg = _load_yaml(CONFIG_DIR / "default.yaml")
    cfg = deep_update(cfg, with_base(path))
    ds = dataset or cfg.get("data", {}).get("dataset")
    ds_cfg = CONFIG_DIR / "datasets" / f"{ds}.yaml"
    if ds and ds_cfg.exists():
        cfg = deep_update(cfg, _load_yaml(ds_cfg))
    cfg.setdefault("data", {})["dataset"] = ds
    overrides = dict(overrides or {})
    skip_tuned = overrides.pop("_no_tuned", False)
    tuned = tuned_path(ds, cfg.get("method"))
    if not skip_tuned and tuned.exists():  # hyper-parameters selected on the TRAIN split by `main.py tune`
        cfg = deep_update(cfg, load_json(tuned)["overrides"])
    if overrides:
        cfg = deep_update(cfg, overrides)
    if "results_dir" not in (overrides or {}) and ds == "synthetic":
        cfg["results_dir"] = "results/demo"
    return cfg


def tuned_path(dataset, method) -> Path:
    return ROOT / "models" / "params" / f"tuned_{dataset}_{method}.json"


def dotted_to_nested(d: Dict[str, Any]) -> Dict:
    out: Dict = {}
    for k, v in d.items():
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def parse_overrides(items) -> Dict:
    """Turn ['tracker.max_age=40', 'rca.use_occ=false'] into a nested dict (values parsed as YAML)."""
    out: Dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Bad override '{it}', expected key.sub=value")
        k, v = it.split("=", 1)
        cur = out
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    return out


def config_hash(obj: Any, n: int = 10) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def save_json(obj: Any, path: Any) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)
    return p


def load_json(path: Any) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def device_info(requested: str = "auto") -> Dict[str, Any]:
    info = {"device": "cpu", "cuda_available": False, "torch": None}
    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if requested in ("auto", "cuda") and info["cuda_available"]:
            info["device"] = "cuda"
    except Exception:
        pass
    return info
