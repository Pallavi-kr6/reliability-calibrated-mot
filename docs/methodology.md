# Methodology

## Problem framing
PS-2 requires identities that survive occlusion and short disappearances **without drifting onto another person**. Both failure
modes are association decisions: a wrong assignment (ID swap / drift) or a missed one (new ID / fragmentation).

## RCA model
Pair features (`algorithm/features.py`): `iou, ctr, tsu=log(frames since last association), d=cosine distance, occ, margin, score`.

```
M    = b_iou*iou + b_ctr*ctr + b_tsu*tsu + b_iou_tsu*iou*tsu
lam  = softplus(g0 + g_occ*occ + g_margin*margin + g_score*score)
z    = b0 + M + lam*(d0 - d); by default d0 is the positive fit-pair median
P    = sigmoid(a*z + b)              (Platt scaling on held-out blocks)
cost = -log P  (forbidden when P < tau)
```
* **Why a margin?** It asks whether appearance separates the *competing candidates of this assignment*: `m_ij = clip(min(min_{j'≠j} d_ij', min_{i'≠i} d_i'j) − d_ij, ±0.3)`
  over gated candidates (no competitor → +0.3). When everybody looks alike, `m≈0` and the learned `g_margin>0` shrinks `lam`, so the decision falls back on motion.
* **Why logistic + Platt?** ~10 parameters, trains in seconds on CPU, interpretable, and the calibrated probability makes "reject" a
  meaningful operation rather than a magic distance threshold.
* **Why fix `d0` by default?** With `d0` optimized at its upper bound, `lam*d0` can act as a reliability-dependent intercept.
  `d0_mode: free_bounded` retains the v0.1 behavior as an explicit ablation; `fixed_median` keeps reliability features tied to the appearance term.

## Label generation (oracle tracks)
Labels must not inherit a baseline's identity drift, so each GT identity gets an oracle track (Kalman + EMA prototype) updated only by its
true detection. For each frame, all (alive oracle track × high-score detection) pairs inside the loose gate get `y = [gt_id(det) == identity]`.
*Known limitation:* oracle tracks are never wrong, so their prototypes are cleaner than online prototypes (distribution shift).
Tracker-in-the-loop relabelling remains future work (rca.relabel_rounds: 0).

## Fitting and calibration
Even 100-frame blocks → logistic fit (L-BFGS-B, analytic gradients, L2 on non-intercept parameters); odd blocks → Platt (a, b).
Calibration quality (ECE 15 bins, Brier, NLL) is evaluated on pairs mined from the **evaluation** split (`main.py calibrate`).
For `gate_mode: far_per_gap`, the held-out odd training blocks provide negative-pair posterior quantiles at `1-target_far`.
The bins are 1, 2–5, 6–15, and 16+ frames since association; a bin uses global `accept_thresh` if it has fewer than 20
positive or fewer than 20 negative calibration pairs. `platt_per_gap` optionally fits separate Platt transforms on those
same blocks; it does not change the gate threshold by itself.

## Hyper-parameter protocol
Every method gets the same treatment: a small grid on the **train** split, tracker in the loop, maximising HOTA (`configs/default.yaml → tune`).
RCA tunes `accept_thresh` and `target_far` on the **train** split only. Tuning on train while RCA's parameters were also fitted on train is mildly optimistic for RCA (≈10 parameters).

## Deviations from the earlier written plan (kept honest)
| Plan | Implementation | Reason |
|---|---|---|
| cost `−logit` | cost `−log P` | non-negative, keeps "maximise number of admissible matches first" behaviour with the BIG-cost trick |
| `tsu = log(1+frames)` | `log(frames since last association)` (0 for a track seen last frame) | same information, zero for continuing tracks |
| E0 = compare with BoxMOT ByteTrack | **not automated**; instead metrics are cross-checked against official TrackEval and the tracker is unit/scenario-tested | BoxMOT was not available in the build sandbox |
| `d0` free | `d0 ∈ [0,1]` in v0.1; v0.2 defaults to positive-fit-pair median | avoid a reliability-dependent intercept; free bounded mode remains available |
| hand-set baseline settings | train-split grid search for all methods | fairness |
| stress windows 100 frames | 100 (real data) / 40 (synthetic demo) | demo sequences are 160 frames |
| one global accept threshold | optional `far_per_gap` threshold from held-out calibration negatives | sparse bins fall back to global tau |
| S2 uses main tracker lifetime | S2 uses `stress.s2_max_age` (90 by default) | allow gaps through 60 frames without changing main tracking |
| fixed S1 windows regardless of sample count | halve window to 40 frames and label sparse bins unreliable | avoid presenting tiny bins as reliable estimates |
| global tau sweep with a per-gap model | add the same offset to all fitted `tau_bins` for the S2 curve | preserve the gap gate while sweeping its operating point |

## Evaluation
`evaluation/metrics.py` re-implements HOTA, CLEAR (MOTA/IDSW) and Identity (IDF1) following TrackEval. On the demo outputs and in
`tests/test_metrics.py::test_matches_official_trackeval` the numbers equal official TrackEval's to two decimals. Pre-processing is
simplified (flag==0 dropped, pedestrian class only, distractor-matched boxes removed). Use `--backend trackeval` for official numbers.

S1 / S2 protocols: see README §16 and `rcamot/evaluation/stress.py` docstring.
