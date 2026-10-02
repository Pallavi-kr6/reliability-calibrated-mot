# third_party

`TrackEval/` — the official evaluation code for HOTA / CLEAR / Identity
(https://github.com/JonathonLuiten/TrackEval, MIT licence, copyright (c) 2020 Jonathon Luiten).
Only the `trackeval/` package, its LICENSE and Readme are included, unmodified. It is used by
`python main.py evaluate --backend trackeval` and by `tests/test_metrics.py` to cross-check the built-in metric
implementation. If the folder is missing, run `python scripts/setup_trackeval.py`.
If you use HOTA, cite: Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking", IJCV 2021.
