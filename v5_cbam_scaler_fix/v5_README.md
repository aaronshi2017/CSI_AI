# v5 — CBAM Attention Encoder + PMI NMSE Scale-Alignment Fix

## Goal

Two independent improvements over v4:
1. Add a CBAM (Convolutional Block Attention Module) to the CsiNet
   encoder, to test whether directing the limited codeword bits toward
   the most informative delay/angular components recovers reconstruction
   quality at high compression ratios (CR=1/16, 1/32) — the regime where
   v4 showed the AI advantage shrinking or reversing.
2. Fix the PMI baseline NMSE metric, which was showing positive dB values
   (physically meaningless — reconstruction error larger than signal
   power) in v4.

## Status: ✅ Complete — both fixes verified working correctly

## What was actually wrong with PMI NMSE (root-cause correction)

Two earlier diagnoses were tried and ruled out before finding the real
cause:

1. **First hypothesis (wrong): wrong-domain scaler.** Suspected the PMI
   path was normalising `H_sf` using the angular-delay domain scaler
   (~33.47) instead of a spatial-frequency domain scaler (~4.46).
   Applying a domain-matched `H_sf_scaler` changed nothing — PMI NMSE
   stayed positive. This makes sense in hindsight: `pmi_type2_reconstruct_sf`
   is a *linear* operator, so NMSE (a ratio of error power to signal
   power) is invariant to any uniform rescaling of its input. The scaler
   substitution could not have fixed the problem by construction.

2. **Second hypothesis (correct): amplitude mismatch in the quantised
   reconstruction.** The 3-bit amplitude / 3-bit phase quantisation in
   the Type II codebook (`pmi_type2_reconstruct_sf`) only optimises beam
   *direction* selection — nothing in the reconstruction guarantees the
   overall magnitude of `Ĥ_pmi` matches the true channel's power. Cosine
   similarity ρ is scale-invariant and was never affected by this (its
   values are identical before/after the fix — direct evidence the bug
   was amplitude-only, not directional). NMSE, however, is *not*
   scale-invariant, so the amplitude mismatch showed up directly as
   inflated (positive) error power.

**Fix applied:** MMSE-optimal per-sample amplitude alignment before
computing PMI NMSE —
```
alpha* = Re(<H_true, H_hat>) / ||H_hat||^2
NMSE   = 10*log10( mean(|alpha*·H_hat - H_true|^2) / mean(|H_true|^2) )
```
This isolates true reconstruction quality from a pure amplitude-scale
artefact of the quantisation scheme — standard practice when evaluating
quantised/reconstructed channel estimates. Applied only to the PMI path;
the AI path is trained end-to-end against the true amplitude and needs
no such correction. **Confirmed correct** by the fact that ρ, AI NMSE,
and BF gain values are bit-for-bit identical before and after the fix —
only PMI NMSE changed, exactly as expected for an amplitude-only bug.

## Final results

| Scenario | CR | AI NMSE | PMI NMSE | NMSE gain | AI ρ | PMI ρ | SGCS gain | Verdict |
|----------|-----|---------|----------|-----------|-------|-------|-----------|---------|
| All CDL | 1/4 | −2.70 dB | −1.74 dB | +0.96 dB | 0.6616 | 0.5114 | +30.7% | ✓ AI wins |
| All CDL | 1/8 | −2.40 dB | −1.74 dB | +0.66 dB | 0.6248 | 0.5114 | +23.2% | ✓ AI wins |
| All CDL | 1/16 | −2.13 dB | −1.74 dB | +0.39 dB | 0.5822 | 0.5114 | +14.5% | ✓ AI wins |
| All CDL | 1/32 | −1.90 dB | −1.74 dB | +0.17 dB | 0.5413 | 0.5114 | +6.1% | ✓ AI wins (narrow) |
| CDL-C | 1/4 | −1.69 dB | −0.87 dB | +0.82 dB | 0.5655 | 0.4116 | +26.2% | ✓ AI wins |
| CDL-C | 1/8 | −1.25 dB | −0.87 dB | +0.38 dB | 0.5017 | 0.4116 | +15.3% | ✓ AI wins |
| CDL-C | 1/16 | −0.90 dB | −0.87 dB | +0.04 dB | 0.4328 | 0.4116 | +3.6% | ✓ AI wins (marginal) |
| CDL-C | 1/32 | −0.65 dB | −0.87 dB | **−0.22 dB** | 0.3683 | 0.4116 | **−7.4%** | **✗ PMI wins** |
| CDL-D | 1/4 | −3.82 dB | −2.61 dB | +1.20 dB | 0.7577 | 0.6112 | +37.7% | ✓ AI wins (strongest) |
| CDL-D | 1/8 | −3.67 dB | −2.61 dB | +1.06 dB | 0.7480 | 0.6112 | +35.2% | ✓ AI wins |
| CDL-D | 1/16 | −3.44 dB | −2.61 dB | +0.83 dB | 0.7316 | 0.6112 | +31.0% | ✓ AI wins |
| CDL-D | 1/32 | −3.23 dB | −2.61 dB | +0.62 dB | 0.7143 | 0.6112 | +26.5% | ✓ AI wins |

NMSE gain and SGCS gain are now consistent in direction at every CR —
both metrics agree on which side wins, which was not true before the
amplitude-alignment fix (some CRs previously showed contradictory
signals between the two metrics).

## Did CBAM help? Partial, measurable improvement — did not flip the
## hardest case to a win

Direct comparison at CDL-C, CR=1/32 (the point v4 flagged as the
weakest for AI):

| Metric | v4 (no CBAM) | v5 (CBAM) | Change |
|--------|--------------|-----------|--------|
| AI ρ | 0.3417 | 0.3683 | **+0.027** |
| SGCS gain vs PMI | ≈ −19.5%† | −7.4% | **+12.1 points** |

† Recomputed from v4's raw ρ values using the same SGCS-gain formula, for
an apples-to-apples comparison; ρ itself is unaffected by the amplitude-
alignment fix (it was already scale-invariant), so this comparison is
valid across versions despite the NMSE-metric fix landing in between.

**Conclusion: CBAM provides a real, non-trivial improvement — it roughly
halved the deficit at the worst-case operating point — but the
underlying problem (NLoS channel too information-rich for 64 codeword
values at CR=1/32) is not fully solved by attention alone.** This matches
expectations from the literature survey done before implementing CBAM:
lightweight channel/spatial attention recovers some quality by directing
bits toward informative components, but closing the gap further likely
needs either:
- A larger-capacity architecture change (e.g. the CRNet-style
  multi-scale/multi-resolution encoder, not yet implemented here), or
- More feedback bits than CR=1/32 allows for this channel type (this is
  a fundamental information-theoretic limit, not purely an architecture
  problem — 24-cluster NLoS channels carry more information than a
  64-value codeword can losslessly represent)

Everywhere else — All CDL combined, CDL-C at CR≤1/16, and all CDL-D
operating points — AI wins on both NMSE and SGCS, with CDL-D (LoS)
showing the strongest and most consistent margin (+26.5% to +37.7% SGCS
gain across every CR tested).

## Files

- `csi_ai_v5.py` — final script: CBAM-equipped `CsiNet`, Type II PMI
  codebook, amplitude-aligned evaluation metrics
- `data/` — CDL-C/D synthetic channel matrices (copied from v4's
  deterministic seed=42 generation, not regenerated)
- `models/` — trained checkpoints (CR=4,8,16,32), `eval_results_v3.json`,
  `comparison_plot_v5.png`

## Reference — v4 baseline (no CBAM, pre-amplitude-alignment-fix)

| CR | AI ρ (v4) | PMI ρ (v4) | SGCS gain (v4) |
|----|-----------|------------|-----------------|
| 1/4 | 0.669 | 0.511 | +32.3% |
| 1/8 | 0.622 | 0.511 | +22.5% |
| 1/16 | 0.585 | 0.511 | +15.0% |
| 1/32 | 0.527 | 0.511 | +3.2% |

Note ρ values changed only marginally between v4 and v5 for "All CDL" —
consistent with CBAM's effect being real but modest, concentrated most
visibly at the hardest operating point (CDL-C, CR=1/32) rather than
uniformly across all scenarios.

## Next steps

- [x] Fix PMI NMSE metric (root cause: amplitude mismatch, not scaler
      domain — see write-up above)
- [x] Retrain with CBAM, confirm partial improvement at CR=1/32
- [ ] (Optional) Try CRNet-style multi-scale encoder as a stronger
      architecture change, if closing the CDL-C/32 gap further is a
      priority
- [ ] Move to `../v6_cost2100/` — switch to measured channel data, where
      the PMI baseline itself should be substantially stronger (real
      channels have exploitable structure the CDL synthetic generator's
      simplified steering-vector model doesn't fully capture), giving a
      fairer and more literature-comparable evaluation overall

## Next version → see `../v6_cost2100/README.md`