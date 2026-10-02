# Interview notes (answers reflect THIS implementation; check your own measured numbers before quoting any)

1. **Why your own tracker instead of modifying BoT-SORT?** Baselines and RCA share one skeleton, so any difference comes from the association decision only.
2. **Why ByteTrack-style two stages?** Low-score boxes recover occluded people; stage 2 uses IoU only because weak boxes give unreliable embeddings.
3. **Failure mode targeted?** Appearance forced into the cost when it cannot separate the competing candidates (similar clothes, uniforms) → identity drift.
4. **Why a candidate-set margin?** Per-detection signals (confidence, overlap) don't say whether appearance discriminates *these* candidates.
5. **Why logistic regression, not a network?** ~10 parameters, seconds to fit, interpretable, calibrates well; limited expressiveness is the price.
6. **How is it calibrated?** Max-likelihood fit, then Platt scaling on held-out blocks; ECE/Brier/NLL on pairs from the evaluation split.
7. **Why softplus on λ and bounded `d0`?** λ>0 so a larger distance never raises match odds; `d0` bounded so reliability features modulate appearance instead of adding free evidence.
8. **Why cost = −log P?** Sum of −log P maximises the product of match posteriors; non-negative so the BIG-cost trick maximises admissible matches first.
9. **How was τ chosen?** Grid on the train split (tracker in the loop); the S2 operating curve shows the whole trade-off.
10. **Where do labels come from?** Oracle tracks from GT (so labels don't inherit a tracker's drift); limitation: oracle prototypes are cleaner than online ones.
11. **Why can IDF1 move while MOTA stays flat?** MOTA is dominated by detection errors; association changes show in AssA/IDF1/IDSW.
12. **Why might a metric get worse?** The gate rejects uncertain matches → more fragmentation (new IDs), visible in S2 and in `after_gap`/`overlapped` ID-switch causes.
13. **What causes an ID switch?** Either a wrong assignment (swap/drift) or a lost track re-created; `failures.py` classifies them.
14. **When do motion and appearance disagree?** After long gaps (stale motion) and in tight, similar-looking groups (ambiguous appearance); `tsu` and `margin` address these.
15. **Why is the long-gap behaviour weak?** True re-associations at long gaps are rare in training pairs, so the calibrated posterior is low; single-τ gating then favours zero wrong-IDs over re-association.
16. **Unseen domains?** Calibration drifts under domain shift; re-fit Platt (cheap) — E8 tests transfer.
17. **Why DanceTrack/uniform data?** Extreme of the hard condition: appearance should be down-weighted, a direct test of the margin hypothesis.
18. **How do you know a gain is real?** Paired bootstrap over windows (≥10), oracle-detection (D0) runs to remove detector confounds, multiple datasets, metrics verified against TrackEval.
19. **Bottleneck / edge deployment?** Embedding extraction; use FP16/ONNX/TensorRT for the embedder, split the assignment into connected components, embed lazily (only ambiguous detections).
20. **With another month?** Tracker-in-the-loop relabelling, gap-aware calibration, a stronger ReID backbone, camera-motion compensation, real-detector runs on DanceTrack.
