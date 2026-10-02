# Experiments

Every experiment answers one question and is a config flip. All values below are produced by the code; unrun cells stay `TO BE MEASURED` / `NOT YET RUN`.

| ID | Question | Command |
|---|---|---|
| E0 | Is the evaluation correct? | `pytest tests/test_metrics.py` (vs official TrackEval) — *BoxMOT comparison is optional and manual* |
| E1 | Where do the baselines stand? | `python main.py main --dataset D` (B0–B3, RCA) |
| E2 | Does calibration/learning alone help? | ablation `rca_const_lambda` |
| E3 | Which reliability signal matters? | ablation `rca_occ_only`, `rca_margin_only`, `rca_occ_margin`, `rca_full` |
| E4 | Does the accept gate help? | ablation `rca_no_gate`; S2 operating curve (τ sweep) |
| E5 | Does posterior-weighted update help? | ablation `rca_fixed_update` |
| E6 | Density stress | `python main.py stress --dataset D` (S1) |
| E7 | Disappearance stress | same command (S2) |
| E8 | Does it transfer? | `python main.py train --dataset mot17` then evaluate `--dataset mot20` with `--set rca.model_path=models/params/rca_mot17.json` |
| E9 | Is the posterior calibrated? | `python main.py calibrate --dataset D` |
| E10 | Runtime | `python main.py benchmark --dataset D` |
| E11 | Backbone sensitivity | rerun `main` with `--embedder colorhist|resnet18|osnet` |

## Decision rules (set before running)
* Claim an improvement only if the paired-bootstrap 95% CI of the per-window/sequence difference excludes zero **and** there are ≥10 units.
* If RCA does not beat B3/B2 anywhere, report that and analyse why (failure strips).
* Report the S2 operating curve, not a single threshold, when comparing false-accept (wrong-ID) against true-accept (re-association).

## Results template (measured values live in `results/`, `results/demo/`)
| Table | File | Status until you run it |
|---|---|---|
| main (HOTA/DetA/AssA/IDF1/MOTA/IDSW/FPS) | `results/main_results.{csv,md}` | NOT YET RUN |
| ablations | `results/ablations.csv` | NOT YET RUN |
| stress S1/S2 | `results/stress.csv` | NOT YET RUN |
| runtime | `results/runtime.csv` | NOT YET RUN |
| calibration | `results/calibration.csv` | NOT YET RUN |
