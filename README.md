# CSI_AI — AI-Based CSI Feedback vs 3GPP PMI Codebooks

Lab project comparing a CsiNet-style autoencoder against 3GPP Type I/II PMI
codebooks for downlink CSI feedback compression, grounded in **3GPP TR 38.843**
(Study on Artificial Intelligence/Machine Learning for NR Air Interface, Rel-18).

## Motivation

In FDD massive MIMO, the UE compresses the downlink channel matrix H and
feeds it back to the gNB over the uplink. Legacy 3GPP codebooks (Type I,
Type II) use fixed DFT beam dictionaries that cannot adapt to the statistics
of real deployed channels. AI/ML autoencoders can learn a channel- and
environment-specific compression mapping instead. TR 38.843 reports
1.4–21.4% SGCS gain over eType II across evaluators in the 3GPP study. This
repo reproduces that comparison across four progressively realistic stages,
from synthetic CDL channels through a Type I baseline up to real measured
COST2100 data with an attention-augmented encoder.

**Full write-up, including background theory, all four stages' results,
a documented training-failure diagnosis and fix, and a discussion of where
AI-based CSI feedback is (and isn't) a good fit across TDD/FDD deployments,
is in [`CSI_Feedback_Progressive_Study.docx`](./CSI_Feedback_Progressive_Study.docx).**

## Project structure

Each version is a self-contained folder — script, its own generated data
(not committed, see below), its own trained models, and a README describing
what changed, what broke, and how it was fixed.

```
CSI_AI/
├── v4_typeII_codebook/     Type II PMI baseline, CDL-C/D synthetic data
├── v5_cbam_scaler_fix/     CBAM attention encoder + PMI amplitude-alignment fix
├── v6_cost2100/            COST2100 measured dataset, weight-decay bug fixed
├── CSI_Feedback_Progressive_Study.docx   Full write-up: theory, results, case study
└── README.md               (this file)
```

Data (`.npy`, `.mat`) and trained checkpoints (`.pth`) are **not** committed
to this repo — see [Regenerating data and models](#regenerating-data-and-models)
below. Each version's script regenerates everything from scratch.

## Common pipeline (every version)

```
1. generate  — produce/load channel matrices H (CDL synthetic or COST2100)
2. train     — train CsiNet autoencoder (UE encoder + gNB decoder) per CR
3. evaluate  — compare AI vs PMI codebook baseline on same test H
```

Run any version with:
```powershell
python csi_ai_vN.py --phase generate
python csi_ai_vN.py --phase train    --epochs 1000
python csi_ai_vN.py --phase evaluate
```

v6 additionally requires `--N_c 1024` on every phase (COST2100's native
bandwidth), and COST2100 `.mat` files must be downloaded manually first
(see `v6_cost2100/README.md`).

## Metrics

| Metric | Meaning | 3GPP reference |
|---|---|---|
| NMSE (dB) | Reconstruction accuracy of Ĥ vs H | — |
| ρ (SGCS) | Spatial Geometry Cosine Similarity — primary KPI | TR 38.843 Table 1 |
| BF gain loss (dB) | Beamforming gain vs perfect SVD precoder | — |
| SGCS % gain | (ρ_AI − ρ_PMI) / (1 − ρ_PMI) × 100 | TR 38.843 reports 1.4–21.4% |

## Results summary

| Stage | Data | Codebook | Encoder | AI ρ range | SGCS gain range | Status |
|---|---|---|---|---|---|---|
| 1 | CDL-C/D synthetic | Type I (1 beam) | Plain CNN | 0.527 – 0.669 | +6% to +32% | ✅ done |
| 2 | CDL-C/D synthetic | Type II (L=4) | Plain CNN | 0.541 – 0.662 | +6% to +31% | ✅ done |
| 3 | CDL-C/D synthetic | Type II (L=4) | CNN + CBAM | 0.541 – 0.662 | +6% to +31%* | ✅ done |
| 4 | COST2100 (real, measured) | Type II (L=4) | CNN + CBAM | 0.802 – 0.966 | +57.5% to +92.7% | ✅ done |

\* Aggregate CDL-C/D numbers barely moved from Stage 2, but CBAM gave a real,
targeted improvement at the single hardest case (CDL-C, CR=1/32, rich NLoS
at extreme compression) — see full write-up, Section 4.2.

The largest jump across all four stages came from switching real measured
data, not from any architecture change (Section 4.3 of the write-up) — the
PMI codebook baseline scored almost identically on synthetic and real data
(ρ≈0.51 vs 0.53), while the AI model's ρ nearly doubled once trained on
real channel structure it could actually learn from.

## Notable finding: a documented training-failure case study

During Stage 4 development, one compression ratio (CR=1/4, the largest
codeword size) collapsed to a degenerate constant output during training,
while the other three converged normally. The full diagnostic process —
including a first hypothesis (BatchNorm channel collapse) that was tested
and ruled out, and the actual root cause (unopposed optimiser weight decay
grinding an already-converged network to zero once training loss
plateaued) — is documented in detail in the write-up, Section 5, along
with the one-line fix and a before/after results comparison. This failure
mode is general to autoencoders whose capacity is large relative to their
task difficulty, and is not specific to this dataset or architecture.

## Where AI-based CSI feedback is (and isn't) a good fit

The write-up (Section 6) also covers when AI genuinely helps versus adds
unnecessary complexity, across TDD and FDD deployments — including why
CSI feedback compression is fundamentally an FDD problem (TDD's SRS +
channel reciprocity already solves this cheaply), where AI still earns
its place in TDD (channel prediction for aging compensation, SRS overhead
reduction for large arrays), why beam-management prediction is arguably
the most defensible AI use case overall (it reduces total measurement
work rather than adding a parallel estimation path), and a staged,
practical deployment path starting with Fixed Wireless Access / static
IoT terminals — where the mobility and model-proliferation objections
that make two-sided AI impractical for mobile handsets simply don't apply.

## Regenerating data and models

```powershell
# v4 / v5 (CDL synthetic — fast, self-contained, uses fixed seed)
cd v4_typeII_codebook   # or v5_cbam_scaler_fix
python csi_ai_v4.py --phase generate
python csi_ai_v4.py --phase train --epochs 1000
python csi_ai_v4.py --phase evaluate

# v6 (COST2100 — requires manual download first, see v6_cost2100/README.md)
cd v6_cost2100
# ... download DATA_Htrainin.mat / DATA_Hvalin.mat / DATA_Htestin.mat into ./data/
python csi_ai_v6.py --phase generate --N_c 1024
python csi_ai_v6.py --phase train --N_c 1024 --epochs 1000
python csi_ai_v6.py --phase evaluate --N_c 1024
```

## Environment

- Python 3.13, PyTorch 2.6.0+cu124
- GPU: NVIDIA RTX 4070 Laptop (8GB VRAM)
- Windows 11, trained locally

## References

- 3GPP TR 38.843, "Study on Artificial Intelligence (AI)/Machine Learning
  (ML) for NR Air Interface," Release 18.
- 3GPP TS 38.214, "NR; Physical layer procedures for data" (Type I/II
  codebook structure).
- C.-K. Wen, W.-T. Shih, S. Jin, "Deep Learning for Massive MIMO CSI
  Feedback," IEEE Wireless Communications Letters, 2018 (CsiNet,
  COST2100 dataset).
- Z. Lu, J. Wang, J. Song, "Multi-resolution CSI Feedback with deep
  learning in Massive MIMO System," arXiv:1910.14322, 2019 (CRNet).