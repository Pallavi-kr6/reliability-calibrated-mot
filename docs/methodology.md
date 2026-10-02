# Methodology

## Problem framing
PS-2 requires identities that survive occlusion and short disappearances **without drifting onto another person**. Both failure
modes are association decisions: a wrong assignment (ID swap / drift) or a missed one (new ID / fragmentation).

## RCA model
Pair features (`algorithm/features.py`): `iou, ctr, tsu=log(frames since last association), d=cosine distance, occ, margin, score`.

```
M    = b_iou*iou + b_ctr*ctr + b_tsu*tsu + b_iou_tsu*iou*tsu
lam  = softplus(g0 + g_occ*occ + g_margin*margin + g_score*score)
z    = b0 + M + lam*(d0 - d),   d0 in [0,1]
P    = sigmoid(a*z + b)              (Platt scaling on held-out blocks)
cost = -log P  (forbidden when P < tau)
```
* **Why a margin?** It asks whether appearance separates the *competing candidates of this assignment*: `m_ij = clip(min(min_{j'≠j} d_ij', min_{i'≠i} d_i'j) − d_ij, ±0.3)`
  over gated candidates (no competitor → +0.3). When everybody looks alike, `m≈0` and the learned `g_margin>0` shrinks `lam`, so the decision falls back on motion.
* **Why logistic + Platt?** ~10 parameters, trains in seconds on CPU, interpretable, and the calibrated probability makes "reject" a
  meaningful operation rather than a magic distance threshold.
* **Why `d0` bounded to [0,1]?** Unbounded, `lam*d0` can act as a free margin/score-dependent intercept, i.e. the reliability features
  would add match evidence instead of modulating the appearance term. (Observed in an early run; fixed.)

## Label generation (oracle tracks)
Labels must not inherit a baseline's identity drift, so each GT identity gets an oracle track (Kalman + EMA prototype) updated only by its
true detection. For each frame, all (alive oracle track × high-score detection) pairs inside the loose gate get `y = [gt_id(det) == identity]`.
*Known limitation:* oracle tracks are never wrong, so their prototypes are cleaner than online prototypes (distribution shift).
A tracker-in-the-loop relabelling round would reduce this; not implemented.

## Fitting and calibration
Even 100-frame blocks → logistic fit (L-BFGS-B, analytic gradients, L2 on non-intercept parameters); odd blocks → Platt (a, b).
Calibration quality (ECE 15 bins, Brier, NLL) is evaluated on pairs mined from the **evaluation** split (`main.py calibrate`).

## Hyper-parameter protocol
Every method gets the same treatment: a small grid on the **train** split, tracker in the loop, maximising HOTA (`configs/default.yaml → tune`).
RCA's only tuned value is the accept threshold τ. Tuning on train while RCA's parameters were also fitted on train is mildly optimistic for RCA (≈10 parameters).

## Deviations from the earlier written plan (kept honest)
| Plan | Implementation | Reason |
|---|---|---|
| cost `−logit` | cost `−log P` | non-negative, keeps "maximise number of admissible matches first" behaviour with the BIG-cost trick |
| `tsu = log(1+frames)` | `log(frames since last association)` (0 for a track seen last frame) | same information, zero for continuing tracks |
| E0 = compare with BoxMOT ByteTrack | **not automated**; instead metrics are cross-checked against official TrackEval and the tracker is unit/scenario-tested | BoxMOT was not available in the build sandbox |
| `d0` free | `d0 ∈ [0,1]` | identifiability (above) |
| hand-set baseline settings | train-split grid search for all methods | fairness |
| stress windows 100 frames | 100 (real data) / 40 (synthetic demo) | demo sequences are 160 frames |

## Evaluation
`evaluation/metrics.py` re-implements HOTA, CLEAR (MOTA/IDSW) and Identity (IDF1) following TrackEval. On the demo outputs and in
`tests/test_metrics.py::test_matches_official_trackeval` the numbers equal official TrackEval's to two decimals. Pre-processing is
simplified (flag==0 dropped, pedestrian class only, distractor-matched boxes removed). Use `--backend trackeval` for official numbers.

S1 / S2 protocols: see README §16 and `rcamot/evaluation/stress.py` docstring.
