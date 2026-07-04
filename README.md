# CSI_AI — AI-Based CSI Feedback vs 3GPP PMI Codebook

Lab project comparing a CsiNet-style autoencoder against 3GPP Type I/II PMI
codebooks for downlink CSI feedback compression, grounded in **3GPP TR 38.843**
(Study on Artificial Intelligence/Machine Learning for NR Air Interface, Rel-18).

## Motivation

In FDD massive MIMO, the UE compresses the downlink channel matrix H and
feeds it back to the gNB over the uplink. Legacy 3GPP codebooks (Type I,
eType II) use fixed DFT beam dictionaries. AI/ML autoencoders can learn a
channel- and environment-specific compression mapping instead. TR 38.843
reports 1.4–21.4% SGCS gain over eType II across evaluators in the 3GPP study.
This repo reproduces that comparison on synthetic and (from v6) measured data.

## Project structure

Each version is a **self-contained folder** — script, its own generated
data, its own trained models, and a README describing what changed and why.

```
CSI_AI/
├── v4_typeII_codebook/     Type II PMI baseline, CDL-C/D synthetic data
├── v5_cbam_scaler_fix/     CBAM attention encoder + NMSE scaler bug fix
├── v6_cost2100/            (planned) COST2100 measured dataset, retrained
└── README.md               (this file)
```

## Common pipeline (every version)

```
1. generate  — produce channel matrices H (CDL synthetic or COST2100 load)
2. train     — train CsiNet autoencoder (UE encoder + gNB decoder) per CR
3. evaluate  — compare AI vs PMI codebook baseline on same test H
```

Run any version with:
```powershell
python csi_ai_vN.py --phase generate
python csi_ai_vN.py --phase train    --epochs 1000
python csi_ai_vN.py --phase evaluate
```

## Metrics

| Metric | Meaning | 3GPP reference |
|---|---|---|
| NMSE (dB) | Reconstruction accuracy of Ĥ vs H | — |
| ρ (SGCS)  | Spatial Geometry Cosine Similarity — primary KPI | TR 38.843 Table 1 |
| BF gain loss (dB) | Beamforming gain vs perfect SVD precoder | — |
| SGCS % gain | (ρ_AI − ρ_PMI) / (1 − ρ_PMI) × 100 | TR 38.843 reports 1.4–21.4% |

## Version history

| Version | Data source | Codebook baseline | Key result | Status |
|---|---|---|---|---|
| v4 | CDL-C/D synthetic | Type II DFT (L=4 beams) | +3–38% SGCS gain | ✅ done, PMI NMSE scaler bug open |
| v5 | CDL-C/D synthetic | Type II DFT | CBAM attention encoder added | 🔄 in progress |
| v6 | COST2100 (measured, indoor 5.3GHz) | Type II DFT | Publication-grade NMSE (~−17dB target) | 📋 planned |

## Environment

- Python 3.13, PyTorch 2.6.0+cu124
- GPU: NVIDIA RTX 4070 Laptop (8GB VRAM)
- Windows 11, trained locally

## References

- 3GPP TR 38.843, "Study on Artificial Intelligence (AI)/Machine Learning
  (ML) for NR Air Interface," Release 18.
- 3GPP TS 38.214, "Physical layer procedures for data" (Type I/II codebooks).
- Wen et al., "Deep Learning for Massive MIMO CSI Feedback," IEEE WCL 2018
  (CsiNet).
- COST2100 channel model dataset — see `v6_cost2100/README.md` for
  download instructions.
