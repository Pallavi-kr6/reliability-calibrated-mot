#!/usr/bin/env python
"""Command line interface for reliability-calibrated-mot.  `python main.py --help`"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rcamot.utils import get_logger, load_config, parse_overrides, set_seed  # noqa: E402

log = get_logger()


def common(p: argparse.ArgumentParser, default_cfg: str = "configs/rca.yaml") -> None:
    p.add_argument("--config", default=default_cfg, help="experiment YAML (merged over configs/default.yaml)")
    p.add_argument("--dataset", choices=["mot17", "mot20", "dancetrack", "synthetic"], help="override data.dataset")
    p.add_argument("--embedder", choices=["colorhist", "resnet18", "osnet"], help="override data.embedder")
    p.add_argument("--det-source", choices=["public", "oracle", "file"], help="override data.det_source")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--only", nargs="+", metavar="SEQ", help="restrict data to the named sequence(s)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], help="embedding device")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VAL",
                   help="generic override, e.g. --set tracker.max_age=40 (repeatable)")


def build_cfg(a):
    ov = parse_overrides(a.sets)
    if a.embedder:
        ov.setdefault("data", {})["embedder"] = a.embedder
    if a.det_source:
        ov.setdefault("data", {})["det_source"] = a.det_source
    if a.seed is not None:
        ov["seed"] = a.seed
    if a.device:
        ov["device"] = a.device
    if a.only:
        ov.setdefault("data", {})["only_sequences"] = a.only
    return load_config(a.config, dataset=a.dataset, overrides=ov), ov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="main.py", description="Reliability-Calibrated Association (RCA) for online multi-person tracking.")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    p = sub.add_parser("prepare", help="verify a dataset layout, or generate the synthetic demo data")
    common(p); p.add_argument("--quick", action="store_true"); p.add_argument("--regen", action="store_true")
    p = sub.add_parser("cache", help="compute + cache detections and appearance embeddings (train+val)")
    common(p); p.add_argument("--force", action="store_true", help="recompute even if the cache fingerprint matches")
    p = sub.add_parser("train", help="mine oracle-track pairs, fit the RCA model, Platt-calibrate, save parameters")
    common(p)
    p = sub.add_parser("track", help="run ONE configured tracker on the eval split and evaluate it")
    common(p)
    p = sub.add_parser("evaluate", help="re-evaluate saved result files (--backend trackeval for the official implementation)")
    common(p); p.add_argument("--backend", choices=["own", "trackeval"], default="own"); p.add_argument("--run-name", help="results sub-folder name (default: config name)")
    p = sub.add_parser("calibrate", help="ECE/Brier/NLL of raw vs Platt-calibrated posteriors on held-out pairs")
    common(p, "configs/calibration.yaml")
    p = sub.add_parser("tune", help="select per-method hyper-parameters on the TRAIN split (grids in configs/default.yaml)")
    common(p); p.add_argument("--methods", nargs="+", choices=["b0", "b1", "b2", "b3", "rca"])
    p = sub.add_parser("main", help="run B0,B1,B2,B3,RCA and write results/main_results.{csv,md}")
    common(p); p.add_argument("--methods", nargs="+", choices=["b0", "b1", "b2", "b3", "rca"])
    p = sub.add_parser("stress", help="S1 density bins (bootstrap CIs) + S2 disappearance injection -> results/stress.csv")
    common(p, "configs/stress.yaml"); p.add_argument("--skip-s2", action="store_true")
    p = sub.add_parser("ablation", help="run the ablation variants -> results/ablations.csv")
    common(p, "configs/ablation.yaml")
    p = sub.add_parser("benchmark", help="runtime profile (tracker latency vs #detections, embedding throughput) -> results/runtime.csv")
    common(p, "configs/stress.yaml")
    p = sub.add_parser("visualize", help="side-by-side videos/GIFs, ID-switch classification and strips")
    common(p); p.add_argument("--seq"); p.add_argument("--methods", nargs="+", default=["b1", "rca"]); p.add_argument("--max-frames", type=int, default=120)
    p = sub.add_parser("report", help="rebuild result tables and refresh the README results section from measured outputs")
    p = sub.add_parser("demo", help="END-TO-END demo on rendered synthetic data (no downloads, CPU only)")
    p.add_argument("--quick", action="store_true"); p.add_argument("--regen", action="store_true"); p.add_argument("--seed", type=int)
    a = ap.parse_args(argv)

    try:
        from rcamot import experiments as ex
        if a.cmd == "report":
            ex.cmd_report(); return 0
        if a.cmd == "demo":
            ex.cmd_demo({"seed": a.seed} if a.seed is not None else {}, quick=a.quick, regen=a.regen); return 0
        cfg, ov = build_cfg(a)
        set_seed(cfg["seed"])
        if a.cmd == "prepare": ex.cmd_prepare(cfg, a.quick, a.regen)
        elif a.cmd == "cache": ex.cmd_cache(cfg, a.force)
        elif a.cmd == "train": ex.cmd_train(cfg)
        elif a.cmd == "track": ex.cmd_track(cfg)
        elif a.cmd == "evaluate": ex.cmd_evaluate(cfg, a.backend, a.run_name)
        elif a.cmd == "calibrate": ex.cmd_calibrate(cfg)
        elif a.cmd == "tune": ex.cmd_tune(cfg["data"]["dataset"], ov, a.methods)
        elif a.cmd == "main": ex.cmd_main(cfg["data"]["dataset"], ov, a.methods)
        elif a.cmd == "stress": ex.cmd_stress(cfg, a.skip_s2)
        elif a.cmd == "ablation": ex.cmd_ablation(cfg)
        elif a.cmd == "benchmark": ex.cmd_benchmark(cfg)
        elif a.cmd == "visualize": ex.cmd_visualize(cfg, a.seq, tuple(a.methods), a.max_frames)
        return 0
    except Exception as e:  # useful message instead of a bare traceback for the expected failure modes
        from rcamot.datasets.registry import DatasetMissingError
        if isinstance(e, (DatasetMissingError, FileNotFoundError, RuntimeError)):
            log.error(f"{type(e).__name__}: {e}")
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
