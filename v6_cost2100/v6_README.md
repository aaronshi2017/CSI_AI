# v6 — COST2100 Measured Dataset + CBAM Attention Encoder

## Goal

Replace the simplified CDL synthetic generator (v4/v5) with the COST2100
measured indoor channel dataset — the standard benchmark used by the
original CsiNet paper (Wen et al., 2018) and nearly all follow-up work
(CRNet, CLNet, DiReNet, etc.) — so results are directly comparable to
published literature. Carries forward the CBAM attention encoder from v5.

## Status: ✅ Ready to run

## Data source

CRNet repo (github.com/Kylin9511/CRNet) → README → Google Drive link.
(Automated `gdown` download is blocked by Google's abuse protection on
this file — downloaded manually via browser instead, then via Baidu
Netdisk mirror.)

Files used (indoor scenario, 5.3 GHz pico-cell):
```
DATA_Htrainin.mat   (~1.4 GB)
DATA_Hvalin.mat     (~0.4 GB)
DATA_Htestin.mat    (~0.3 GB)
```
Copied from external drive into `.\data\` before running (see Setup below).

Also downloaded but **not used** in this version: `DATA_Htrainout.mat` /
`DATA_Hvalout.mat` / `DATA_Htestout.mat` (outdoor scenario) and the
`*Fin_all` / `*Fout_all` full-bandwidth variants — kept on the external
drive for potential future outdoor comparison.

## What changed vs v5

1. **New `load_cost2100()` function** replaces `run_generate()`:
   - Loads `.mat` files via `scipy.io.loadmat`, key `"HT"`
   - Transposes (N, 32, 32, 2) → (N, 2, 32, 32) to match the existing
     `CsiNet` input convention
   - Already normalised to roughly [-1, 1] by the dataset authors — no
     additional AI-side scaler needed (`scaler = 1.0`, saved to
     `scalers.json` as `{"COST2100_indoor": 1.0}`)
   - Reconstructs `H_sf` (spatial-frequency domain) for the PMI baseline
     path by zero-padding the 32 delay taps to the dataset's native
     `N_c = 1024` subcarriers and applying the inverse 2D DFT
     (`angular_delay_to_sf`, same function as v4/v5)

2. **CBAM attention encoder carried forward unchanged from v5** — channel
   + spatial attention block inserted between the two `ConvBN` layers in
   `CsiNet.encode()`. Not removed for this version: CBAM is an
   architecture change, COST2100 is a data change, and the two are
   independent — no reason to drop one when adding the other.
   - **Trade-off accepted:** this version changes two variables at once
     (dataset + architecture), so any NMSE/ρ improvement over v4 cannot be
     cleanly attributed to "COST2100 alone" vs "COST2100 + CBAM together."
     If clean attribution is needed later, rerun with CBAM commented out
     (revert `self.enc` to `nn.Sequential(ConvBN(2,16), ConvBN(16,8))`)
     as a controlled comparison.

3. **Type II PMI codebook unchanged from v4/v5** (`pmi_type2_reconstruct_sf`,
   `L=4` beams, 3-bit amplitude/phase quantisation) — same fixed baseline,
   now tested against real measured channels instead of synthetic CDL.

## Critical setup note: `--N_c 1024` required on every phase

COST2100's native bandwidth is 1024 subcarriers, vs 64 for the CDL
synthetic generator used in v4/v5. `args.N_c` defaults to 64 in the CLI.
**Every command must explicitly pass `--N_c 1024`**, or the AI path's
`angular_delay_to_sf()` call in `evaluate_both_paths` will zero-pad to the
wrong width (64 instead of 1024), producing a shape mismatch against the
PMI ground truth (which `load_cost2100()` correctly built at N_c=1024).

## Setup

```powershell
cd "C:\Users\roger\Downloads\My development\CSI_AI\v6_cost2100"
New-Item -ItemType Directory -Path ".\data" -Force

# Copy indoor scenario files from external drive
Copy-Item "E:\BaiduNetdiskDownload\COST2100\DATA_Htrainin.mat" ".\data\"
Copy-Item "E:\BaiduNetdiskDownload\COST2100\DATA_Hvalin.mat"   ".\data\"
Copy-Item "E:\BaiduNetdiskDownload\COST2100\DATA_Htestin.mat"  ".\data\"

Get-ChildItem ".\data" | Select-Object Name, @{N="MB";E={[math]::Round($_.Length/1MB,1)}}
```

## Run sequence

```powershell
# 1. Load COST2100 (fast — just parsing .mat files, no simulation)
& "C:/Users/roger/AppData/Local/Microsoft/WindowsApps/python3.13.exe" `
  ".\csi_ai_v6.py" --phase generate --N_c 1024

# 2. Train — retraining required, CDL-trained weights from v4/v5 do NOT
#    transfer (different channel statistics). ~45min-1hr on RTX 4070 Laptop.
& "C:/Users/roger/AppData/Local/Microsoft/WindowsApps/python3.13.exe" `
  ".\csi_ai_v6.py" --phase train --N_c 1024 --epochs 1000

# 3. Evaluate
& "C:/Users/roger/AppData/Local/Microsoft/WindowsApps/python3.13.exe" `
  ".\csi_ai_v6.py" --phase evaluate --N_c 1024
```

## Known cosmetic items (non-blocking)

- `plot_comparison()` title/legend still reads "CDL-C / CDL-D" — should be
  updated to "COST2100 Indoor" before sharing plots externally, but does
  not affect correctness of the numbers.
- The per-scenario evaluation loop in `run_evaluate()` looks for
  `H_test_CDL{sc}.npy` files, which `load_cost2100()` doesn't produce —
  it will simply find nothing and skip, leaving only the single
  "All CDL" (really "All COST2100") result row. This is expected: v6 has
  one dataset, not five CDL scenarios like v4/v5.

## Expected results (from published CsiNet/CRNet papers, same dataset)

| CR   | CsiNet NMSE (published) | eType II baseline NMSE (approx) |
|------|--------------------------|-----------------------------------|
| 1/4  | ~ -17 dB                 | ~ -7 to -10 dB                   |
| 1/8  | ~ -12 dB                 | ~ -5 dB                           |
| 1/16 | ~ -9 dB                  | ~ -4 dB                           |
| 1/32 | ~ -6 dB                  | ~ -2 dB                           |

This is a substantial jump from the ~ -3 dB NMSE seen on the CDL synthetic
generator in v4/v5 — expected, since COST2100 is real measured data with
genuine sparsity structure that both CsiNet and the Type II codebook can
exploit, unlike the simplified CDL steering-vector model.

## Reference — v4/v5 CDL synthetic baseline (for comparison)

| CR   | AI ρ (v4, CDL synthetic) | PMI ρ (v4) | SGCS gain (v4) |
|------|---------------------------|------------|-----------------|
| 1/4  | 0.669                     | 0.511      | +32.3%          |
| 1/8  | 0.622                     | 0.511      | +22.5%          |
| 1/16 | 0.585                     | 0.511      | +15.0%          |
| 1/32 | 0.527                     | 0.511      | +3.2%           |

v6 target: absolute NMSE numbers in the published range above, plus SGCS
gain over Type II PMI that should land closer to the 1.4–21.4% range
reported in 3GPP TR 38.843 (the CDL synthetic data in v4/v5 inflated this
gain because the PMI codebook was mismatched to the synthetic steering
vectors, not because the AI was genuinely that much better).

## Files

- `csi_ai_v6.py` — full script: `load_cost2100()`, CBAM-equipped `CsiNet`,
  Type II PMI codebook, spatial-frequency metrics, evaluation + plotting
- `data/` — COST2100 `.mat` files (not committed to git — see `.gitignore`)
  plus generated `.npy`/`scalers.json` after running `--phase generate`
- `models/` — trained checkpoints + eval results + comparison plot,
  created by `--phase train` / `--phase evaluate`

## Next steps

- [ ] Run the three-phase sequence above and record actual results
- [ ] Update plot titles from "CDL-C / CDL-D" to "COST2100 Indoor"
- [ ] (Optional) Controlled comparison: rerun with CBAM removed to isolate
      its contribution from the dataset switch
- [ ] (Optional) Repeat with outdoor scenario (`DATA_H*out.mat`, already
      downloaded) for an indoor/outdoor generalisation comparison