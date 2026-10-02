#!/usr/bin/env python
"""One command for the core evaluation pipeline.

  python scripts/reproduce.py --quick          # ~1-3 min, CPU, no downloads: synthetic DEMO data, validates the whole stack
  python scripts/reproduce.py --full           # benchmark datasets that are present under data/ (MOT17 / MOT20 / DanceTrack)
  python scripts/reproduce.py --full --datasets mot17 --embedder colorhist   # weights-free CPU run on MOT17

Steps: validate environment -> validate dataset -> (re)build cache -> train+calibrate RCA -> tune on TRAIN split ->
track B0..B3/RCA -> calibration -> stress (S1/S2) -> ablation -> runtime -> figures/tables -> README results section.
Nothing is hard-coded: all numbers come from the runs. Unrun experiments stay `NOT YET RUN`.
"""
import argparse
import importlib
import platform
import sys

import _bootstrap  # noqa: F401
from rcamot import experiments as ex
from rcamot.datasets.registry import DatasetMissingError
from rcamot.utils import device_info, get_logger, load_config, set_seed

log = get_logger()


def validate_environment() -> None:
    log.info(f"Python {platform.python_version()} on {platform.system()}")
    if sys.version_info < (3, 9):
        raise SystemExit("Python >= 3.9 is required")
    missing = []
    for m in ("numpy", "scipy", "cv2", "yaml", "pandas", "matplotlib", "PIL"):
        try:
            mod = importlib.import_module(m)
            log.info(f"  {m:11s} {getattr(mod, '__version__', 'ok')}")
        except Exception:
            missing.append(m)
    if missing:
        raise SystemExit(f"Missing packages: {missing}. Run: pip install -r requirements.txt")
    info = device_info("auto")
    log.info(f"Device: {info['device'].upper()} | CUDA available: {info['cuda_available']} | torch: {info['torch']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["mot17", "mot20", "dancetrack"], choices=["mot17", "mot20", "dancetrack"])
    ap.add_argument("--embedder", choices=["colorhist", "resnet18", "osnet"], help="override the dataset default embedder")
    ap.add_argument("--det-source", choices=["public", "oracle", "file"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    validate_environment()
    if a.quick:
        ex.cmd_demo({"seed": a.seed}, quick=True, regen=False)
        log.info("REPRODUCE --quick OK. Outputs: results/demo/ (synthetic DEMO data, not a benchmark)")
        return 0

    ran = 0
    for ds in a.datasets:
        ov = {"seed": a.seed}
        if a.embedder:
            ov["data"] = {"embedder": a.embedder}
        if a.det_source:
            ov.setdefault("data", {})["det_source"] = a.det_source
        cfg = load_config("configs/rca.yaml", dataset=ds, overrides=ov)
        set_seed(a.seed)
        try:
            ex.cmd_prepare(cfg)                 # validates the layout (raises DatasetMissingError with instructions)
        except DatasetMissingError as e:
            log.warning(f"Skipping {ds}: {e}")
            continue
        try:
            ex.cmd_cache(cfg)
            ex.cmd_train(cfg)
            ex.cmd_calibrate(cfg)
            ex.cmd_tune(ds, ov)
            ex.cmd_main(ds, ov)
            ex.cmd_stress(load_config("configs/stress.yaml", dataset=ds, overrides=ov))
            ex.cmd_ablation(load_config("configs/ablation.yaml", dataset=ds, overrides=ov))
            ex.cmd_benchmark(load_config("configs/stress.yaml", dataset=ds, overrides=ov))
            ran += 1
        except (RuntimeError, FileNotFoundError) as e:
            log.error(f"{ds}: {e}")
    ex.cmd_report()
    if not ran:
        log.error("No benchmark dataset could be processed. Download one first (scripts/download_mot17.py) or run --quick.")
        return 2
    log.info("REPRODUCE --full finished; see results/*.csv, results/main_results.md, results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
