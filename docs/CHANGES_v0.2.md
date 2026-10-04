# Changes in v0.2

## 1. Summary

- RCA remains a single reliability-calibrated association method; the MOT17 results do not show a decisive tracking gain.
- Candidate-pair diagnostics contradict the idea that colorhist is near chance on the mined MOT17 pairs.
- Training now fixes `d0` to the median positive fit-pair distance by default, avoiding the previous bound-saturated value.
- A held-out, per-gap false-accept gate and corrected S1/S2 protocols are available by default.
- The matched-false-accept criterion was not met; the main-metric margin was met narrowly.

## 2. Evidence that motivated the change

The pre-change benchmark values below were captured by `scripts/make_changes_tables.py` from the generated README results block into [`results/diagnostics/before_main_results.csv`](../results/diagnostics/before_main_results.csv). The after values come from [`results/main_results.csv`](../results/main_results.csv); the full generated comparison is [`results/diagnostics/changes_tables.md`](../results/diagnostics/changes_tables.md).

| Method | Before HOTA / AssA / IDF1 / IDSW | After HOTA / AssA / IDF1 / IDSW |
|---|---|---|
| B0 | 49.1 / 58.6 / 55.8 / 248 | 49.10 / 58.56 / 55.82 / 248 |
| B1 | 48.6 / 57.5 / 55.0 / 223 | 48.55 / 57.52 / 55.04 / 223 |
| B2 | 48.8 / 58.1 / 55.4 / 219 | 48.84 / 58.12 / 55.39 / 219 |
| B3 | 49.3 / 59.0 / 56.2 / 248 | 49.27 / 59.05 / 56.24 / 248 |
| RCA | 48.7 / 57.7 / 55.2 / 243 | 49.12 / 58.69 / 55.95 / 242 |

The held-out-pair calibration CSV records ECE 0.0158 before Platt scaling and 0.0020 after it (`results/calibration.csv`). The pre-change tracked `models/params/rca_mot17.json` had `d0=1.0`, at the old upper bound; the CLI-generated v0.2 model now records `d0=0.0201`, `d0_mode=fixed_median`, and per-gap thresholds in that same file. Train-only tuning selected `accept_thresh=0.0` and `target_far=0.02` (`models/params/tuned_mot17_rca.json`).

New val-pair measurements are in [`results/diagnostics/appearance_mot17_colorhist.csv`](../results/diagnostics/appearance_mot17_colorhist.csv) and [`results/diagnostics/gate_mot17.csv`](../results/diagnostics/gate_mot17.csv): appearance-only AUC of `-d` is 0.996 overall and 0.955 for pairs with a lost track; the lost-pair positive rate is 0.77%. For gaps 6–15, the median positive posterior is 0.0125 and the positive rejection rate is 70.6%; for 16+, they are 0.0020 and 52.6%. Negative acceptance in both bins remains near the configured two-percent target. The computed matched-FAR and ablation tables are in [`results/diagnostics/changes_tables.md`](../results/diagnostics/changes_tables.md).

The corrected S2 run used 230 valid cases at gap 15 and 194 at gap 30. At the selected gap-aware thresholds, RCA re-associated 69.6% / 49.0% at those gaps, with wrong-ID rates 2.2% / 3.6%. Baseline wrong-ID rates (8.3–9.8% at gaps 15/30) lie outside the measured RCA sweep range (maximum 2.2% / 3.6%); matched-FAR interpolation is therefore NaN and no matched-FAR performance claim is made. S2 gap 60 is now measurable under a 90-frame lifetime; 134–135 cases were valid in the generated run. S1 reduced its window from 100 to 50 frames, yielding 13–14 windows per crowding bin.

## 3. Diagnosis: H-A..H-D verdicts with the supporting file paths

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H-A: colorhist carries almost no real pedestrian identity information | **Contradicted for mined MOT17 val pairs.** The `-d` AUC is 0.996 overall and 0.955 for lost-track pairs. This is candidate-pair evidence and does not establish online tracking benefit. | `results/diagnostics/appearance_mot17_colorhist.csv` |
| H-B: bound-saturated `d0` acts as reliability-dependent intercept evidence | **Supported as an identifiability concern.** The pre-change tracked model had `d0=1.0`; fixed-median training stores `d0=0.0201`. The `rca_free_d0` ablation scored HOTA 49.26 vs 49.11 for full RCA, so fixing it did not improve this val result. | `models/params/rca_mot17.json`, `models/params/ablation_mot17_rca_free_d0.json`, `results/ablations.csv` |
| H-C: long-gap positives have low posterior because of pair base rate, so a global threshold rejects reappearances | **Supported.** The 6–15 and 16+ bins contain 85 and 38 positives among 14,077 and 23,370 pairs; their median positive posteriors are 0.0125 and 0.0020. Positive rejection remains 70.6% and 52.6%, respectively. The gate keeps negative acceptance near 2%, but matched-FAR interpolation is NaN because baseline wrong-ID rates are outside the RCA curve. | `results/diagnostics/gate_mot17.csv`, `results/stress.csv` (`S2_matched_far`) |
| H-D: oracle-track fit pairs differ from online pairs | **Open.** The current diagnostics measure mined oracle-track pairs; no online-pair distribution comparison or tracker-in-the-loop relabelling was completed. | `rcamot/training/mine_pairs.py`, `docs/methodology.md` |

## 4. Changes

| File | Change | Reason | Test |
|---|---|---|---|
| `rcamot/models/rca.py` | Fixed-parameter fitting; `d0_mode`, `tau_bins`, optional `platt_bins`; legacy JSON defaults; per-gap threshold helpers. | Address d0 identifiability and represent gap-aware acceptance. | `tests/test_calibration.py` |
| `rcamot/training/train_rca.py` | Fix d0 to the positive fit-pair median; fit held-out Platt calibration and false-accept quantiles; optional per-gap Platt fits before deriving thresholds. | Keep gate calibration on training calibration blocks. | `tests/test_calibration.py`, pipeline tests |
| `rcamot/algorithm/association.py` | Apply the model's per-gap accept threshold when configured. | Make the saved gate active during online assignment. | `tests/test_association.py` |
| `rcamot/evaluation/stress.py`, `rcamot/experiments.py` | S2 max-age override; S1 adaptive windowing/reliability flags; shifted per-gap tau curve and matched-FAR interpolation. | Correct the stress protocol and compare at equal wrong-ID rate. | `tests/test_stress_protocols.py` |
| `main.py`, `rcamot/inference/pipeline.py` | Add `--only SEQ [SEQ ...]` and thread sequence filtering through config and detection-set selection. | Run selected sequences across commands. | `tests/test_pipeline.py` |
| `configs/default.yaml`, `configs/ablation.yaml` | Add d0/gate settings, target-FAR grid, stress settings, and requested ablations. | Expose reproducible defaults and comparisons. | Config-loaded by full test suite |
| `scripts/diagnose_appearance.py`, `scripts/diagnose_gate.py` | Generate rank-AUC and gate-rejection diagnostics from mined val pairs. | Verify H-A and H-C before changing behavior. | Ran on MOT17/colorhist; outputs cited above |
| `scripts/make_changes_tables.py` | Generate before/after, matched-FAR, and ablation tables from result CSVs; write `NOT YET RUN` for absent results and `NaN` for out-of-range interpolation. | Keep report tables sourced from code outputs. | Ran; generated `results/diagnostics/changes_tables.md` |
| `docs/methodology.md`, `README.md`, `CHANGELOG.md`, `rcamot/__init__.py` | Record protocol/model deviations, limitations, v0.2 notes, and version `0.2.0`. | Keep user-facing documentation aligned with implementation. | Reviewed against generated files |

## 5. New/changed config keys

| Key | Default | Meaning |
|---|---|---|
| `rca.d0_mode` | `fixed_median` | Fix d0 to the median positive fit-pair cosine distance; `free_bounded` restores the old bounded optimization. |
| `rca.gate_mode` | `far_per_gap` | Use a global threshold or per-gap thresholds fitted from calibration negatives. |
| `rca.target_far` | `0.02` | Target negative candidate acceptance rate; also included in the train-only tune grid. |
| `rca.platt_per_gap` | `false` | Fit per-gap Platt transforms when each bin has at least 20 examples of both classes. |
| `rca.relabel_rounds` | `0` | Reserved; tracker-in-the-loop relabelling remains unimplemented. |
| `stress.s2_max_age` | `90` | Track lifetime used only by S2 disappearance injection. |
| `stress.min_windows_per_bin` | `10` | Minimum S1 windows per bin before the bin is marked reliable. |

`stress.tau_sweep` now applies a common offset to the learned per-gap thresholds, preserving the fitted gate shape during S2 operating-curve evaluation.

## 6. Behavioural differences and anything that is NOT backward compatible

Existing JSON model files still load. Files without `d0_mode`, `tau_bins`, or `platt_bins` receive legacy defaults; with no stored bin thresholds, the gate falls back to the configured global `accept_thresh`. CLI flags remain compatible, and `--only` is optional.

New training defaults are behaviorally different: d0 is fixed to the positive fit-pair median and association uses `far_per_gap`. Set `rca.d0_mode: free_bounded` and `rca.gate_mode: global` to request the v0.1 model/gate behavior. S2 results change because its tracker now keeps tracks up to `stress.s2_max_age`; main-table tracking still uses `tracker.max_age`. Sparse S1 runs may use smaller windows and carry a `reliable` flag.

The S2 tau-sweep values are interpreted as common threshold offsets from the selected global `accept_thresh` baseline over all `tau_bins`; this replaces the prior scalar-threshold sweep for the operating curve. The CLI syntax and output CSV location are unchanged.

## 7. How to reproduce (exact commands)

Run from the repository root with MOT17 installed under `data/MOT17/train` and the project dependencies available:

```powershell
python main.py cache --dataset mot17 --embedder colorhist
python main.py train --dataset mot17 --embedder colorhist
python main.py tune --dataset mot17 --embedder colorhist --methods rca
python scripts/diagnose_appearance.py --dataset mot17 --embedder colorhist
python scripts/diagnose_gate.py --dataset mot17 --embedder colorhist
python main.py main --dataset mot17 --embedder colorhist
python main.py stress --dataset mot17 --embedder colorhist --set "stress.gap_lengths=[15,30,60]" --set "stress.gap_seeds=[0,1]" --set stress.n_gaps=20
python main.py ablation --dataset mot17 --embedder colorhist
python scripts/make_changes_tables.py
python main.py report
$env:PYTHONPATH=(Resolve-Path .venv/Lib/site-packages).Path
python -m pytest -p no:cacheprovider -ra
```

This host had no torch installation, so the optional `resnet18` appearance diagnostic was not run. If CPU torch is available, run `python main.py cache --dataset mot17 --embedder resnet18`, then repeat the appearance diagnostic with `--embedder resnet18` and use a model trained with that embedder.

## 8. Results before vs after: tables generated by a script

The following tables are emitted by `python scripts/make_changes_tables.py` from the generated result files. The complete output, including `NOT YET RUN` rows for MOT20 and DanceTrack, is [`results/diagnostics/changes_tables.md`](../results/diagnostics/changes_tables.md).

| Method | Before HOTA | After HOTA | Before IDF1 | After IDF1 | Before IDSW | After IDSW |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 49.1 | 49.10 | 55.8 | 55.82 | 248 | 248 |
| B1 | 48.6 | 48.55 | 55.0 | 55.04 | 223 | 223 |
| B2 | 48.8 | 48.84 | 55.4 | 55.39 | 219 | 219 |
| B3 | 49.3 | 49.27 | 56.2 | 56.24 | 248 | 248 |
| RCA | 48.7 | 49.12 | 55.2 | 55.95 | 243 | 242 |

S2 matched-FAR values are NaN because the baseline wrong-ID rates are outside the measured RCA sweep range. Each gap 15/30 cell has at least 100 valid cases; no extrapolation is reported.

| Baseline | Gap | Baseline wrong-ID | Baseline re-association | RCA at matched FAR | Delta |
|---|---:|---:|---:|---:|---:|
| B0 | 15 | 0.0957 | NaN | NaN |
| B0 | 30 | 0.0979 | NaN | NaN |
| B1 | 15 | 0.0913 | NaN | NaN |
| B2 | 15 | 0.0826 | NaN | NaN |
| B3 | 15 | 0.0965 | NaN | NaN |
| B1 | 30 | 0.0876 | NaN | NaN |
| B2 | 30 | 0.0928 | NaN | NaN |
| B3 | 30 | 0.0979 | NaN | NaN |

The measured val ablations (`results/ablations.csv`) were: full RCA HOTA 49.12; free d0 49.29; global gate 42.36; far-per-gap 49.12; far-per-gap with per-bin Platt 49.05. In this run the global threshold selected on train was 0.0, so the global-gate and no-gate variants coincide. Neither d0 fixing nor gap calibration produced a clear headline-metric gain; the per-bin Platt fit had little effect.

## 9. Success-criteria verdict (met / not met, per criterion)

1. **Met.** The complete suite reports 75 passed; `test_matches_official_trackeval` passed and checks the expected two-decimal parity.
2. **Primary not met / not comparable; secondary met.** At 15 and 30 frames, valid gap counts were 230 and 194. Baseline wrong-ID rates are outside the RCA curve range, so matched-FAR results are NaN; criterion 2 is not established as met. Main-table RCA HOTA is 49.12 versus best baseline B3 at 49.27 (0.14 lower); IDF1 is 55.95 versus 56.24 (0.29 lower), within the pre-set 0.3 margin.
3. **Met.** The RCA threshold and target-FAR grid were selected on the train split; val was used only for diagnostics and final reporting. No threshold was changed after inspecting val results.

## 10. Known limitations and next steps

- The near-chance colorhist hypothesis was contradicted on mined pairs, but the diagnostic does not test online embedding drift. AUC also does not imply a tracker-level gain.
- Gap-aware thresholds keep candidate false acceptance low, but do not recover baseline re-association at matched wrong-ID rates in this run.
- Sparse long-gap positives remain a calibration limitation; optional per-gap Platt scaling reduced HOTA/IDF1 in the measured ablation.
- Oracle-track vs online-pair distribution shift remains unmeasured. `rca.relabel_rounds` stays at 0; tracker-in-the-loop relabelling is future work.
- MOT20, DanceTrack, and resnet18 were not run here. No benchmark claim is made for those datasets/backends.
- The ablation is single-seed MOT17 val evidence, and the primary S2 comparison is a controlled disappearance protocol, not a natural-occlusion benchmark.
