# Demo

`python main.py demo --quick` (≈1–2 min) or `python main.py demo` (≈5 min) — CPU only, no downloads.

The demo **renders its own synthetic scenes** (`rcamot/datasets/synthetic.py`) into `data/synthetic/` (not shipped; regenerated in seconds)
and runs the *same* code path as the real datasets: cache → train/calibrate RCA → tune on train → B0–B3 + RCA → stress → ablation → runtime → visualise → report.

**These are synthetic DEMO results, not benchmark results.** Do not quote them as such.

Scenes: a scripted `syn_story` (two near-identical blue-shirt people cross with near-total overlap; a wall occluder; one person leaves for 15 frames and returns) and
procedural crowd sequences of growing density with diverse → uniform clothing.

Where to look afterwards
* `results/demo/main_results.md` – B0–B3 vs RCA
* `results/demo/figures/syn_story_b1_vs_rca.gif` – side-by-side overlay (IDs are colour-coded)
* `results/demo/figures/s2_operating_curve.png`, `s1_idf1_vs_crowding.png`, `reliability.png`, `ablation.png`
* `results/demo/failures/idsw_by_cause.csv` and `switch_*.png`

A copy of the overlay GIF from the build run is in `demo/syn_story_b1_vs_rca.gif`.

## First successful checkpoint (what a working install prints)

`python main.py demo --quick` (or `python scripts/reproduce.py --quick`) should print, in order, lines of this shape
(numbers are from the build run on Linux/CPU/Python 3.12 and differ slightly across platforms and between `--quick` and the full demo):

```
Device: CPU | CUDA available: False | torch: None
[synthetic] rendering train/syn_train_a (8 people, 70 frames)         # data generated in seconds
[train] 6733 pairs (pos rate 0.245); params = b0=..., b_iou=..., g_margin=..., d0=...
[train] Platt a=... b=...; saved .../models/params/rca_synthetic.json
[calibration/uncalibrated] ECE 0.0221  Brier 0.0131  NLL 0.0595  (n=7025)
[calibration/platt] ECE 0.0068  Brier 0.0116  NLL 0.0501  (n=7025)     # Platt ECE < uncalibrated ECE
[tune] b0 ... [tune] b1 ... [tune] b2 ... [tune] b3 ... [tune] rca: best on train HOTA = ... with {'rca.accept_thresh': ...}
[b0] ... [b1] ... [b2] ... [b3] ... [rca] ... per-sequence HOTA/AssA/IDF1/MOTA/IDSW lines
[S2] b0 k=10: n=... success=... wrong=... new=... lost=0   (and the same for b1..b3, rca, rca_tau=...)
Rendered {'mp4': '.../syn_story_b1_vs_rca.mp4', 'gif': '.../syn_story_b1_vs_rca.gif'}
=== demo finished in ~50s; see results/demo/ ===                        # `--quick`; the full demo takes ~4-5 min on one CPU core
REPRODUCE --quick OK. Outputs: results/demo/ (synthetic DEMO data, not a benchmark)
```

Success criteria: exit code 0, no `ERROR`/`Traceback`, and `results/demo/main_results.md` contains five rows (b0–b3, rca).
`python -m pytest` should report all tests passing (about 5 s).
