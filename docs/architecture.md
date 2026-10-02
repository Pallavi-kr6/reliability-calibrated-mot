# Architecture

```
main.py                      CLI (prepare, cache, train, tune, track, main, evaluate, calibrate, stress, ablation, benchmark, visualize, report, demo)
rcamot/
  utils.py                   paths (always relative to the project root), config merge, seeds, logging, hashing
  experiments.py             command implementations (orchestration only; no algorithms)
  datasets/                  mot_io (MOT format), registry (splits), synthetic (renderer), download (stdlib downloader)
  inference/                 embedder (colorhist | resnet18 | osnet), cache (DetSet + fingerprints), pipeline (glue)
  algorithm/                 geometry, kalman, track (lifecycle + EMA prototype), features (pair features + margin),
                             association (B0-B3 / RCA costs, Hungarian + accept gate), tracker (two-stage online tracker)
  models/rca.py              RCA model, L-BFGS fit, Platt scaling, calibration metrics
  training/                  mine_pairs (oracle-track labels), train_rca (fit + calibrate + calibration eval)
  evaluation/                metrics (HOTA/CLEAR/IDF1), runner, stress (S1/S2), bootstrap, trackeval_backend
  visualization/             plots, draw (videos/GIFs), failures (ID-switch causes + strips)
configs/                     default.yaml + one file per experiment + datasets/<name>.yaml
third_party/TrackEval        official metric code (MIT), used for cross-checking
```

## Data flow
1. **cache** – detections (public / oracle / file) + appearance embeddings are computed once per (dataset, sequence, det source,
   embedder, frame range) and stored as `.npz` with a JSON fingerprint. Every tracker config reads these arrays.
2. **train** – `mine_pairs` replays GT identities as *oracle tracks* and emits (features, label) pairs with the exact same
   `compute_pair_features` the online tracker uses (no train/serve skew in the feature code).
3. **tune** – per-method grid search on the train split (tracker in the loop), results saved to `models/params/tuned_<dataset>_<method>.json`
   and applied automatically by `load_config` (explicit `--set` overrides win).
4. **track/main/stress/ablation** – `run_tracker` over cached detections → MOT result files + `metrics.json`.
5. **report** – tables and the README results section are rebuilt from `metrics.json`/CSVs only.

## Online tracker (per frame)
Kalman predict → high/low split → stage 1 (confirmed+lost × high dets; method-specific cost, Hungarian, accept gate) →
tentative × remaining high (IoU) → stage 2 (tracked × low dets, IoU only) → lifecycle → new tracks.

## Cost of one frame
Features and margin are O(N·M) (row/column top-2); Hungarian is O(n³) worst case on the *gated* cost matrix. Embeddings are the
dominant cost on real data (computed once per detection; cached offline for experiments).
