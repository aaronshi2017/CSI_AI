# v4 — Type II PMI Codebook Baseline

## Summary
CsiNet autoencoder vs 3GPP Type II DFT codebook, evaluated on CDL-C/CDL-D
synthetic channels in the spatial-frequency domain.

## Status
- AI beats PMI baseline: +3% to +38% SGCS gain across CR 1/4 to 1/32
- Known issue: PMI NMSE still shows positive dB (scaler bug not fully fixed)
- Known issue: CBAM attention module added but not retrained — no effect yet

## Files
- csi_ai_v4.py       — main script (generate/train/evaluate)
- models/            — trained CsiNet checkpoints (CR=4,8,16,32) + eval results
- data/              — generated CDL-C/D channel matrices (H_train/val/test)

## Results (SGCS gain over PMI, All CDL)
| CR   | AI rho | PMI rho | Gain   |
|------|--------|---------|--------|
| 1/4  | 0.669  | 0.511   | +32.3% |
| 1/8  | 0.622  | 0.511   | +22.5% |
| 1/16 | 0.585  | 0.511   | +15.0% |
| 1/32 | 0.527  | 0.511   | +3.2%  |

## Known limitations
1. PMI NMSE positive (scaler domain mismatch between H_a and H_sf) — needs fix
2. CDL-C at CR=1/32 shows AI losing to PMI (-11.9% in earlier version)
3. CBAM attention added to encoder but requires retraining to take effect

## Next steps → see v5_cbam_scaler_fix/
