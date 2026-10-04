"""Command implementations behind main.py (prepare/cache/train/track/evaluate/calibrate/stress/ablation/benchmark/
visualize/report/demo)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .datasets.registry import DatasetMissingError
from .evaluation.metrics import METRIC_KEYS
from .evaluation.runner import METHOD_CONFIGS, method_cfg, read_predictions, run_dir, run_experiment
from .evaluation.stress import add_bins, choose_s1_window, matched_far_table, offset_tau_bins, s1_tables, s1_windows, s2_table
from .inference.pipeline import ensure_model, get_detsets, run_tracker
from .models.rca import RCAModel
from .utils import deep_update, device_info, ensure_dir, get_logger, load_config, resolve_path, save_json, set_seed

log = get_logger()
BENCH_DATASETS = ("mot17", "mot20", "dancetrack")
METHODS = ["b0", "b1", "b2", "b3", "rca"]


def banner(cfg: Dict) -> None:
    info = device_info(cfg.get("device", "auto"))
    log.info(f"Device: {info['device'].upper()} | CUDA available: {info['cuda_available']} | torch: {info['torch']}")
    log.info(f"Dataset: {cfg['data']['dataset']} | det source: {cfg['data']['det_source']} | embedder: {cfg['data']['embedder']} "
             f"| seed: {cfg['seed']} | results: {cfg['results_dir']}")


def fig_dir(cfg) -> Path:
    return ensure_dir(Path(cfg["results_dir"]) / "figures")


# ------------------------------------------------------------------ prepare / cache / train
def cmd_prepare(cfg: Dict, quick: bool = False, regen: bool = False) -> None:
    ds = cfg["data"]["dataset"]
    if ds == "synthetic":
        from .datasets.synthetic import generate_demo_dataset
        root = resolve_path(cfg["data"]["root"]) / "synthetic"
        if root.exists() and not regen:
            log.info(f"Synthetic data already present at {root} (use --regen to rebuild).")
            return
        generate_demo_dataset(root, quick=quick, seed=cfg["seed"])
        log.info(f"Synthetic DEMO data written to {root} (not a benchmark).")
        return
    from .datasets.registry import list_sequences
    for split in ("train", "val"):
        specs = list_sequences(ds, split, cfg["data"]["root"], cfg["data"].get("mot17_detector", "FRCNN"),
                               cfg["data"].get("only_sequences"))
        log.info(f"{ds}/{split}: {len(specs)} sequences, {sum(s.frame_end - s.frame_start + 1 for s in specs)} frames")
        for s in specs:
            log.info(f"  {s.name}: frames {s.frame_start}-{s.frame_end}  gt={'ok' if s.gt_path.exists() else 'MISSING'}  "
                     f"det={'ok' if s.det_path.exists() else 'absent'}")


def cmd_cache(cfg: Dict, force: bool = False) -> None:
    banner(cfg)
    t0 = time.time()
    for split in ("train", "val"):
        dss = get_detsets(cfg, split, force=force)
        log.info(f"[cache] {split}: {sum(len(d.frames) for d in dss)} detections over {len(dss)} sequences")
    log.info(f"[cache] done in {time.time() - t0:.1f}s -> {cfg['cache_dir']}")


def cmd_train(cfg: Dict) -> RCAModel:
    banner(cfg)
    from .training.train_rca import train_rca
    model, _ = train_rca(cfg)
    return model


def cmd_track(cfg: Dict) -> Dict:
    banner(cfg)
    res = run_experiment(cfg)
    c = res["combined"]
    log.info(f"[{cfg['name']}] COMBINED  HOTA {c['HOTA']:.2f}  AssA {c['AssA']:.2f}  IDF1 {c['IDF1']:.2f}  MOTA {c['MOTA']:.2f}  IDSW {c['IDSW']}"
             f"  | tracker {res['timing'].get('tracker_fps', float('nan')):.0f} FPS (excl. detector/embedding)")
    log.info(f"Results written to {run_dir(cfg)}")
    return res


def cmd_evaluate(cfg: Dict, backend: str = "own", run_name: Optional[str] = None) -> Dict:
    """Re-evaluate saved result files (no tracking)."""
    from .evaluation.metrics import combine, evaluate, public
    from .inference.cache import gt_eval_rows
    banner(cfg)
    split = cfg["data"]["eval_split"]
    dss = get_detsets(cfg, split)
    rdir = run_dir(cfg, run_name)
    preds = read_predictions(rdir, dss)
    if backend == "trackeval":
        from .evaluation.trackeval_backend import evaluate_with_trackeval
        out = evaluate_with_trackeval([d.spec for d in dss], preds, cfg)
        save_json(out, rdir / "metrics_trackeval.json")
        log.info(f"TrackEval COMBINED: {out['combined']}")
        return out
    per = {d.spec.name: evaluate(gt_eval_rows(d.spec), preds[d.spec.name], (d.spec.frame_start, d.spec.frame_end), cfg["eval"]["iou_thr"]) for d in dss}
    comb = combine(list(per.values()))
    out = {"combined": public(comb), "per_sequence": {k: public(v) for k, v in per.items()}, "backend": "own"}
    save_json(out, rdir / "metrics_reeval.json")
    log.info(f"COMBINED: { {k: round(v, 2) if isinstance(v, float) else v for k, v in out['combined'].items()} }")
    return out


def cmd_calibrate(cfg: Dict) -> Dict:
    banner(cfg)
    from .training.train_rca import evaluate_calibration
    model = ensure_model(cfg)
    res = evaluate_calibration(cfg, model, Path(cfg["results_dir"]) / "calibration.csv", fig_dir(cfg) / "reliability.png")
    for k in ("uncalibrated", "platt"):
        log.info(f"[calibration/{k}] ECE {res[k]['ECE']:.4f}  Brier {res[k]['Brier']:.4f}  NLL {res[k]['NLL']:.4f}  (n={res[k]['n']})")
    return res


# ------------------------------------------------------------------ multi-method runs
def run_all(dataset: str, overrides: Dict, methods: List[str], split: Optional[str] = None):
    """Track all methods on shared cached detections. Returns (cfgs, results, detsets, models)."""
    cfgs = {m: method_cfg(m, dataset, overrides) for m in methods}
    base = next(iter(cfgs.values()))
    split = split or base["data"]["eval_split"]
    dss = get_detsets(base, split)
    models = {}
    if "rca" in cfgs:
        models["rca"] = ensure_model(cfgs["rca"])
    results = {m: run_experiment(c, split, dss, models.get(m)) for m, c in cfgs.items()}
    return cfgs, results, dss, models


def cmd_tune(dataset: str, overrides: Dict, methods: Optional[List[str]] = None) -> Dict:
    """Grid-search each method's key hyper-parameters on the TRAIN split (tracker-in-the-loop) and persist the winner.
    Every method gets the same treatment, so the comparison is not tilted by hand-picked baseline settings."""
    import itertools
    from .utils import dotted_to_nested, tuned_path
    methods = methods or METHODS
    out = {}
    for m in methods:
        ov = deep_update(overrides, {"_no_tuned": True})
        base = method_cfg(m, dataset, ov)
        grid = base["tune"]["grids"].get(m)
        if not grid:
            continue
        split = base["tune"]["split"]
        dss = get_detsets(base, split)
        model = ensure_model(base) if m == "rca" else None
        keys = list(grid)
        best, table = None, []
        for combo in itertools.product(*[grid[k] for k in keys]):
            c = deep_update(base, dotted_to_nested(dict(zip(keys, combo))))
            candidate_model = model
            if m == "rca" and "rca.target_far" in keys:
                from .training.train_rca import train_rca
                candidate_model, _ = train_rca(c)
            r = run_experiment(c, split, dss, candidate_model, write=False)["combined"]
            score = r[base["tune"]["metric"]]
            table.append({**dict(zip(keys, combo)), base["tune"]["metric"]: score, "IDF1": r["IDF1"], "IDSW": r["IDSW"]})
            if best is None or score > best[0]:
                best = (score, dict(zip(keys, combo)))
        save_json({"method": m, "dataset": dataset, "split": split, "metric": base["tune"]["metric"], "best_score": best[0],
                   "overrides": dotted_to_nested(best[1]), "table": table}, tuned_path(dataset, m))
        if m == "rca" and "rca.target_far" in keys:
            from .training.train_rca import train_rca
            train_rca(deep_update(base, dotted_to_nested(best[1])))
        log.info(f"[tune] {m}: best on {split} {base['tune']['metric']} = {best[0]:.2f} with {best[1]}  ({len(table)} configs)")
        out[m] = best
    return out


def cmd_main(dataset: str, overrides: Dict, methods: Optional[List[str]] = None) -> pd.DataFrame:
    methods = methods or METHODS
    cfgs, res, dss, _ = run_all(dataset, overrides, methods)
    banner(cfgs[methods[0]])
    return build_main_results(cfgs[methods[0]]["results_dir"], [dataset])


def cmd_stress(cfg: Dict, skip_s2: bool = False) -> None:
    banner(cfg)
    S = cfg["stress"]
    methods = list(S["methods"])
    ov = {"data": cfg["data"], "seed": cfg["seed"], "results_dir": cfg["results_dir"], "cache_dir": cfg["cache_dir"]}
    cfgs, results, dss, models = run_all(cfg["data"]["dataset"], ov, methods)
    preds = {m: results[m]["_preds"] for m in methods}
    s1_window, s1_enough = choose_s1_window(dss, S["window"], S["n_bins"], S.get("min_windows_per_bin", 10))
    if s1_window != S["window"]:
        log.warning(f"S1 window reduced from {S['window']} to {s1_window} frames to target {S.get('min_windows_per_bin', 10)} windows per bin.")
    if not s1_enough:
        log.warning("S1 remains sparse at the 40-frame minimum; all bins will be marked unreliable.")
    df = add_bins(s1_windows(preds, dss, s1_window, cfg["eval"]["iou_thr"]), S["n_bins"])
    df.attrs["min_windows_per_bin"] = S.get("min_windows_per_bin", 10) if s1_enough else 10 ** 9
    target = "rca" if "rca" in methods else methods[-1]
    agg, diff = s1_tables(df, S["reference"], target, S["n_boot"], cfg["seed"])
    out = Path(cfg["results_dir"])
    ensure_dir(out)
    df.to_csv(resolve_path(out / "stress_s1_windows.csv"), index=False)
    parts = [agg.assign(table="S1_bin_means")]
    if len(diff):
        parts.append(diff.assign(table="S1_paired_bootstrap"))
    if not skip_s2:
        s2 = s2_table(cfgs, models, dss, S["gap_lengths"], S["n_gaps"], S["gap_seeds"])
        parts.append(s2.assign(table="S2_gap_injection"))
        from .visualization.plots import plot_s2, plot_s2_curve
        plot_s2(s2, fig_dir(cfg) / "s2_gap_injection.png")
        if "rca" in cfgs and S.get("tau_sweep"):
            # Sweep a common additive offset over the learned per-gap thresholds, preserving the
            # proposed gate shape while moving along its re-association / wrong-ID operating curve.
            import copy
            base_tau = float(cfgs["rca"]["rca"]["accept_thresh"])
            curve_cfgs, curve_models = {}, {}
            for tau in S["tau_sweep"]:
                key = f"rca_tau={tau}"
                curve_cfgs[key] = deep_update(cfgs["rca"], {"rca": {"accept_thresh": tau, "gate_mode": "far_per_gap"}})
                curve_models[key] = copy.deepcopy(models["rca"])
                curve_models[key].tau_bins = offset_tau_bins(models["rca"].tau_bins, tau - base_tau)
            curve = s2_table(curve_cfgs, curve_models, dss, S["gap_lengths"], S["n_gaps"], S["gap_seeds"])
            parts.append(curve.assign(table="S2_operating_curve"))
            matched = matched_far_table(s2[s2.method != "rca"], curve)
            parts.append(matched.assign(table="S2_matched_far"))
            plot_s2_curve(s2, curve, fig_dir(cfg) / "s2_operating_curve.png")
    pd.concat(parts, ignore_index=True).to_csv(resolve_path(out / "stress.csv"), index=False)
    from .visualization.plots import plot_s1
    plot_s1(agg, fig_dir(cfg) / "s1_idf1_vs_crowding.png", "IDF1")
    log.info(f"Stress results -> {out / 'stress.csv'}  (windows: {len(df['w0'].unique())} per method)")
    print(agg.round(2).to_string(index=False))
    if len(diff):
        print(diff.round(3).to_string(index=False))


def cmd_ablation(cfg: Dict) -> pd.DataFrame:
    banner(cfg)
    from .training.train_rca import train_rca
    base = method_cfg("rca", cfg["data"]["dataset"], {"data": cfg["data"], "seed": cfg["seed"], "results_dir": cfg["results_dir"],
                                                       "cache_dir": cfg["cache_dir"]})
    dss = get_detsets(base, base["data"]["eval_split"])
    full = ensure_model(base)
    rows = []
    for v in cfg["ablation"]["variants"]:
        c = deep_update(base, v["overrides"])
        c["name"] = v["name"]
        if v.get("refit"):
            c["rca"]["model_path"] = f"models/params/ablation_{base['data']['dataset']}_{v['name']}.json"
            model, _ = train_rca(c)
        else:
            model = full
        r = run_experiment(c, base["data"]["eval_split"], dss, model)
        rows.append({"variant": v["name"], **{k: r["combined"][k] for k in METRIC_KEYS}, "dataset": base["data"]["dataset"]})
    df = pd.DataFrame(rows)
    out = resolve_path(Path(cfg["results_dir"]) / "ablations.csv")
    df.to_csv(out, index=False)
    from .visualization.plots import plot_ablation
    plot_ablation(df, fig_dir(cfg) / "ablation.png")
    print(df.round(2).to_string(index=False))
    return df


def cmd_benchmark(cfg: Dict) -> pd.DataFrame:
    banner(cfg)
    ov = {"data": cfg["data"], "seed": cfg["seed"], "results_dir": cfg["results_dir"], "cache_dir": cfg["cache_dir"]}
    methods = cfg.get("stress", {}).get("methods", METHODS)
    cfgs, results, dss, _ = run_all(cfg["data"]["dataset"], ov, methods)
    bins = [(0, 10), (10, 30), (30, 60), (60, 10 ** 6)]
    rows = []
    for m in methods:
        # re-time with a fresh run so that file IO / evaluation never enters the timing
        model = ensure_model(cfgs[m]) if m == "rca" else None
        pf = []
        for ds in dss:
            _, st = run_tracker(ds, cfgs[m], model)
            pf += st["per_frame"]
        n = np.array([a for a, _ in pf])
        ms = np.array([b for _, b in pf]) * 1000
        for lo, hi in bins:
            sel = (n >= lo) & (n < hi)
            if sel.sum() == 0:
                continue
            rows.append({"component": "tracker", "method": m, "dets_bin": f"{lo}-{hi if hi < 10 ** 6 else 'inf'}", "frames": int(sel.sum()),
                         "mean_ms": float(ms[sel].mean()), "p95_ms": float(np.percentile(ms[sel], 95)), "fps_equiv": float(1000 / ms[sel].mean())})
    # embedding throughput
    from .inference.embedder import get_embedder
    emb = get_embedder(cfg["data"], cfg.get("device", "auto"))
    crops = [(np.random.default_rng(i).random((160, 64, 3)) * 255).astype(np.uint8) for i in range(128)]
    emb.embed(crops[:8])
    t0 = time.perf_counter()
    emb.embed(crops)
    dt = (time.perf_counter() - t0) / len(crops)
    rows.append({"component": f"embedder:{emb.name}", "method": "-", "dets_bin": "per crop", "frames": len(crops),
                 "mean_ms": dt * 1000, "p95_ms": float("nan"), "fps_equiv": float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(resolve_path(Path(cfg["results_dir"]) / "runtime.csv"), index=False)
    from .visualization.plots import plot_runtime
    plot_runtime(df[df.component == "tracker"], fig_dir(cfg) / "runtime.png")
    print(df.round(3).to_string(index=False))
    return df


def cmd_visualize(cfg: Dict, seq: Optional[str] = None, methods=("b1", "rca"), max_frames: int = 120) -> None:
    banner(cfg)
    from .visualization.draw import render_comparison
    from .visualization.failures import failure_report
    ov = {"data": cfg["data"], "seed": cfg["seed"], "results_dir": cfg["results_dir"], "cache_dir": cfg["cache_dir"]}
    cfgs, results, dss, _ = run_all(cfg["data"]["dataset"], ov, list(dict.fromkeys(list(methods) + ["rca"])))
    preds = {m: results[m]["_preds"] for m in cfgs}
    pick = [d for d in dss if (seq is None or d.spec.name == seq)]
    if seq is None:
        pick = [next((d for d in dss if "story" in d.spec.name), dss[-1])]
    for d in pick:
        fr = range(d.spec.frame_start, min(d.spec.frame_end, d.spec.frame_start + max_frames - 1) + 1)
        out = render_comparison(d.spec, {m: preds[m][d.spec.name] for m in methods}, fig_dir(cfg) / f"{d.spec.name}_{'_vs_'.join(methods)}", fr)
        log.info(f"Rendered {out}")
    summ = failure_report(dss, {m: preds[m] for m in preds}, Path(cfg["results_dir"]) / "failures",
                          reference=methods[0], target=methods[-1])
    log.info(f"ID-switch causes:\n{summ.to_string()}")


# ------------------------------------------------------------------ reporting
def _fmt(v):
    return "NOT YET RUN" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.1f}" if isinstance(v, float) else str(v))


def build_main_results(results_dir, datasets) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for m in METHODS:
            p = resolve_path(results_dir) / "runs" / ds / m / "metrics.json"
            if p.exists():
                import json
                r = json.loads(p.read_text())
                rows.append({"dataset": ds, "method": m, **{k: r["combined"][k] for k in METRIC_KEYS},
                             "tracker_fps": r["timing"].get("tracker_fps"), "det_source": r["det_source"], "embedder": r["embedder"]})
            else:
                rows.append({"dataset": ds, "method": m, **{k: None for k in METRIC_KEYS}, "tracker_fps": None, "det_source": None, "embedder": None})
    df = pd.DataFrame(rows)
    out = resolve_path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "main_results.csv", index=False)
    md = ["| dataset | method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | tracker FPS | det source | embedder |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        md.append(f"| {r.dataset} | {r.method} | " + " | ".join(_fmt(getattr(r, k)) for k in METRIC_KEYS) + f" | {_fmt(r.tracker_fps)} | {_fmt(r.det_source)} | {_fmt(r.embedder)} |")
    (out / "main_results.md").write_text("\n".join(md) + "\n")
    return df


def cmd_report() -> None:
    """Rebuild results tables and refresh the auto-generated README section from measured outputs only."""
    build_main_results("results", list(BENCH_DATASETS))
    build_main_results("results/demo", ["synthetic"])
    blocks = ["#### Benchmark results (public datasets) — `results/main_results.md`\n", (resolve_path("results/main_results.md")).read_text(),
              "\n#### Synthetic DEMO results (NOT a benchmark) — `results/demo/main_results.md`\n", resolve_path("results/demo/main_results.md").read_text()]
    for name, title in (("results/demo/ablations.csv", "Synthetic DEMO ablations"), ("results/demo/calibration.csv", "Synthetic DEMO calibration")):
        p = resolve_path(name)
        if p.exists():
            d = pd.read_csv(p)
            blocks.append(f"\n#### {title} — `{name}`\n\n" + d.round(3).to_markdown(index=False) + "\n" if _has_tabulate() else f"\n#### {title} — `{name}`\n\n```\n{d.round(3).to_string(index=False)}\n```\n")
    s2p = resolve_path("results/demo/stress.csv")
    if s2p.exists():
        st = pd.read_csv(s2p)
        g = st[st.table == "S2_gap_injection"]
        if len(g):
            t = g.pivot(index="method", columns="gap_frames", values=["reassoc_rate", "wrong_id_rate"]).round(3)
            t.columns = [f"{a}@{int(b)}f" for a, b in t.columns]
            blocks.append("\n#### Synthetic DEMO — S2 disappearance injection (rates) — `results/demo/stress.csv`\n\n```\n" + t.to_string() + "\n```\n")
        b = st[st.table == "S1_bin_means"]
        if len(b):
            blocks.append("\n#### Synthetic DEMO — S1 crowding bins (means over windows) — `results/demo/stress.csv`\n\n```\n" +
                          b[["bin", "method", "n_windows", "kappa", "HOTA", "AssA", "IDF1", "IDSW"]].round(2).to_string(index=False) + "\n```\n")
    cp = resolve_path("results/demo/failures/idsw_by_cause.csv")
    if cp.exists():
        blocks.append("\n#### Synthetic DEMO — ID switches by cause — `results/demo/failures/idsw_by_cause.csv`\n\n```\n" +
                      pd.read_csv(cp).to_string(index=False) + "\n```\n")
    text = "\n".join(blocks)
    readme = resolve_path("README.md")
    if readme.exists():
        s = readme.read_text(encoding="utf-8")
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in s and b in s:
            s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
            readme.write_text(s, encoding="utf-8")
            log.info("README results section refreshed from measured outputs.")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ demo
def cmd_demo(overrides: Dict, quick: bool = False, regen: bool = False) -> None:
    """Whole pipeline on synthetic data: generate -> cache -> train -> calibrate -> track B0..B3/RCA -> stress ->
    ablation -> benchmark -> visualise -> report. Synthetic DEMO only; benchmark tables stay NOT YET RUN."""
    ov = deep_update({"data": {"dataset": "synthetic"}}, overrides or {})
    if quick:
        ov = deep_update(ov, {"stress": {"n_boot": 300, "gap_lengths": [10, 30], "gap_seeds": [0], "n_gaps": 5, "tau_sweep": [0.05, 0.2]}})
    cfg = load_config("configs/rca.yaml", dataset="synthetic", overrides=ov)
    set_seed(cfg["seed"])
    t0 = time.time()
    log.info("=== SYNTHETIC DEMO (rendered locally; not benchmark data) ===")
    cmd_prepare(cfg, quick=quick, regen=regen)
    cmd_cache(cfg)
    cmd_train(cfg)
    cmd_calibrate(cfg)
    cmd_tune("synthetic", ov)
    cmd_main("synthetic", ov)
    scfg = load_config("configs/stress.yaml", dataset="synthetic", overrides=ov)
    cmd_stress(scfg)
    acfg = load_config("configs/ablation.yaml", dataset="synthetic", overrides=ov)
    cmd_ablation(acfg)
    cmd_benchmark(scfg)
    cmd_visualize(cfg)
    cmd_report()
    log.info(f"=== demo finished in {time.time() - t0:.0f}s; see results/demo/ ===")
