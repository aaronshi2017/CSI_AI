"""
csi_ai.py  —  v5.0
═══════════════════════════════════════════════════════════════════════════════
CSI Feedback AI vs PMI Codebook Lab Project
─────────────────────────────────────────────────────────────────────────────
v5 additional fixes vs v4:
use CBAM: add a CBAM-style channel+spatial attention block before the FC layer. This directly targets "spend the few bits I have on what matters":
 multi-rate/multi-scale encoder (CRNet-style). Replace your single-path ConvBN(2,16)→ConvBN(16,8) with two parallel branches (3×3 and 1×9/9×1 kernels) concatenated before the FC layer, so the network captures both localized and long-range delay/angular correlations at low bit budgets.
No retraining needed.  No re-generation needed.
DROP-IN REPLACEMENT for run_train() in csi_ai_v5.py

Two additions:
  1. Smart early stopping — only resets patience counter on MEANINGFUL
     improvement (> min_delta), not on noise-level fluctuations. Stops
     training sooner once genuinely plateaued instead of chasing 0.01 dB
     changes for 50 extra epochs.
  2. Skip-if-checkpoint-exists — if csinet_CR{CR}.pth already exists,
     skip training that CR entirely (safe to interrupt/resume across
     sessions). Use --force_retrain to override and retrain anyway.

Also add this to the argparse section (near the other ap.add_argument lines):
    ap.add_argument("--force_retrain", action="store_true",
                    help="Retrain even if a checkpoint already exists")
    ap.add_argument("--min_delta", type=float, default=0.02,
                    help="Minimum NMSE improvement (dB) to reset patience counter")
"""

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, argparse, json

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR  = r"./data"
MODEL_DIR = r"./models"
os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class CBAM(nn.Module):
    """Lightweight channel + spatial attention, insert before enc_fc."""
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.ca_fc = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid())
        self.sa_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        # Channel attention
        avg = x.mean(dim=[2,3])
        ca  = self.ca_fc(avg).unsqueeze(-1).unsqueeze(-1)
        x   = x * ca
        # Spatial attention
        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([avg_sp, max_sp], dim=1)))
        return x * sa

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

CDL_PARAMS = {
    "A": dict(n_clusters=23, delay_spread=1000e-9, angle_spread_az=65, k_factor=0,  label="CDL-A (Rich NLoS)"),
    "B": dict(n_clusters=23, delay_spread=300e-9,  angle_spread_az=50, k_factor=0,  label="CDL-B (Moderate NLoS)"),
    "C": dict(n_clusters=24, delay_spread=300e-9,  angle_spread_az=35, k_factor=0,  label="CDL-C (Low-scatter NLoS)"),
    "D": dict(n_clusters=13, delay_spread=30e-9,   angle_spread_az=15, k_factor=13, label="CDL-D (Strong LoS)"),
    "E": dict(n_clusters=15, delay_spread=30e-9,   angle_spread_az=10, k_factor=22, label="CDL-E (Very strong LoS)"),
}

def generate_cdl_H(n_samples=50000, N_t=32, N_c=64, scenario="C",
                   fc=3.5e9, snr_db=20, seed=42):
    """Returns (N, N_t, N_c) complex — spatial-frequency domain."""
    rng = np.random.default_rng(seed)
    p   = CDL_PARAMS[scenario]
    n_c, ds, as_, kf = p["n_clusters"], p["delay_spread"], p["angle_spread_az"], p["k_factor"]
    BW, df = 100e6, 100e6 / N_c
    H = np.zeros((n_samples, N_t, N_c), dtype=complex)
    for s in range(n_samples):
        delays  = rng.exponential(ds, size=n_c); delays -= delays.min()
        aods    = rng.laplace(0, as_ / np.sqrt(2), size=n_c) * np.pi / 180
        powers  = np.exp(-delays / ds)
        if kf > 0: powers[0] += kf * powers.sum()
        powers /= powers.sum()
        for ci in range(n_c):
            n_per_pol = N_t // 2
            a_h = np.exp(1j * 2*np.pi * 0.5 * np.arange(n_per_pol) * np.sin(aods[ci]))
            co  = rng.standard_normal(2) @ [1, 1j]
            a   = np.concatenate([a_h, a_h * co / np.abs(co)])
            hf  = np.exp(-1j * 2*np.pi * np.arange(N_c) * df * delays[ci])
            g   = (np.sqrt(powers[ci]) if (ci == 0 and kf > 0)
                   else (rng.standard_normal() + 1j*rng.standard_normal())
                        * np.sqrt(powers[ci] / 2))
            H[s] += np.outer(a, hf) * g
        np_ = np.mean(np.abs(H[s])**2) / (10**(snr_db/10))
        H[s] += (rng.standard_normal((N_t, N_c))
                 + 1j*rng.standard_normal((N_t, N_c))) * np.sqrt(np_ / 2)
    return H

def preprocess(H_raw, N_p=32):
    """
    Spatial-freq H  →  normalised angular-delay H_a  (for AI training).
    Returns (H_norm float32 (N,2,N_t,N_p),  scaler float).
    """
    H_a   = np.fft.ifft(H_raw, axis=2)
    H_a   = np.fft.fft(H_a,   axis=1)
    H_a   = H_a[:, :, :N_p]
    H_2ch = np.stack([H_a.real, H_a.imag], axis=1).astype(np.float32)
    scaler = float(np.max(np.abs(H_2ch)) + 1e-8)
    return H_2ch / scaler, scaler

def run_generate(args):
    print("=" * 60)
    print("STEP 1 — Generating CDL channel matrices")
    print("=" * 60)
    scenarios = args.cdl.split(",") if args.cdl != "all" else list(CDL_PARAMS.keys())
    all_train, all_val, all_test = [], [], []
    n_train, n_val, n_test = 100000, 30000, 20000
    scalers = {}

    for sc in scenarios:
        print(f"\n  Scenario CDL-{sc} ({CDL_PARAMS[sc]['label']})...")
        n_total = n_train + n_val + n_test
        H_raw   = generate_cdl_H(n_total, N_t=args.N_t, N_c=args.N_c,
                                  scenario=sc, snr_db=args.snr)
        H_norm, scaler = preprocess(H_raw, N_p=args.N_p)
        scalers[sc] = scaler

        all_train.append(H_norm[:n_train])
        all_val.append(  H_norm[n_train:n_train+n_val])
        all_test.append( H_norm[n_train+n_val:])

        # Save raw spatial-frequency test set for PMI evaluation (Fix 8)
        H_sf_test = H_raw[n_train+n_val:].astype(np.complex64)
        np.save(f"{DATA_DIR}/H_sf_test_CDL{sc}.npy", H_sf_test)
        np.save(f"{DATA_DIR}/H_test_CDL{sc}.npy", H_norm[n_train+n_val:])
        print(f"    Generated {n_total} samples, scaler={scaler:.4f}")

    H_train = np.concatenate(all_train)
    H_val   = np.concatenate(all_val)
    H_test  = np.concatenate(all_test)
    np.save(f"{DATA_DIR}/H_train.npy", H_train)
    np.save(f"{DATA_DIR}/H_val.npy",   H_val)
    np.save(f"{DATA_DIR}/H_test.npy",  H_test)

    # Combined spatial-freq test set
    H_sf_all = np.concatenate([
        np.load(f"{DATA_DIR}/H_sf_test_CDL{sc}.npy") for sc in scenarios], axis=0)
    np.save(f"{DATA_DIR}/H_sf_test_all.npy", H_sf_all)

    # Fix 9: save scalers with indent, verify non-empty
    with open(f"{DATA_DIR}/scalers.json", "w") as f:
        json.dump(scalers, f, indent=2)
    print(f"\nScalers saved: {scalers}")
    print(f"Saved: train {H_train.shape}, val {H_val.shape}, test {H_test.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AI MODEL  (Path A — unchanged from v2)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBN(nn.Module):
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ic,oc,k,1,p,bias=False),
                               nn.BatchNorm2d(oc), nn.LeakyReLU(0.1, True))
    def forward(self, x): return self.b(x)

class RefineBlock(nn.Module):
    def __init__(self, ch=2):
        super().__init__()
        self.c1 = ConvBN(ch, ch)
        self.c2 = nn.Sequential(nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch))
        self.a  = nn.LeakyReLU(0.1, True)
    def forward(self, x): return self.a(self.c2(self.c1(x)) + x)

class CsiNet(nn.Module):
    def __init__(self, N_t=32, N_p=32, CR=8):
        super().__init__()
        self.M  = 2 * N_t * N_p // CR
        self.N_t, self.N_p = N_t, N_p
        # self.enc     = nn.Sequential(ConvBN(2, 16), ConvBN(16, 8))
        self.enc = nn.Sequential(ConvBN(2,16), CBAM(16), ConvBN(16,8))
        self.enc_fc  = nn.Linear(8 * N_t * N_p, self.M)
        self.enc_sig = nn.Sigmoid()
        self.dec_fc  = nn.Linear(self.M, 2 * N_t * N_p)
        self.refine  = nn.Sequential(RefineBlock(2), RefineBlock(2))
        self.dec_out = nn.Sequential(nn.Conv2d(2, 2, 3, 1, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc_sig(self.enc_fc(self.enc(x).flatten(1)))

    def decode(self, s):
        x = self.dec_fc(s).view(-1, 2, self.N_t, self.N_p)
        return self.dec_out(self.refine(x))

    def forward(self, x): return self.decode(self.encode(x))

def nmse_loss(H_hat, H_true):
    mse = ((H_hat - H_true)**2).mean(dim=[1,2,3])
    pwr = (H_true**2).mean(dim=[1,2,3])
    return (mse / (pwr + 1e-8)).mean()

# def run_train(args):
    print("=" * 60)
    print("STEP 2 — Training CsiNet (Path A)")
    print("=" * 60)
    H_tr = torch.from_numpy(np.load(f"{DATA_DIR}/H_train.npy"))
    H_va = torch.from_numpy(np.load(f"{DATA_DIR}/H_val.npy"))
    tr_dl = DataLoader(TensorDataset(H_tr, H_tr), batch_size=512,
                       shuffle=True,  num_workers=0, pin_memory=False)
    va_dl = DataLoader(TensorDataset(H_va, H_va), batch_size=512,
                       shuffle=False, num_workers=0, pin_memory=False)

    def nmse_db_t(H_hat, H_true):
        return 10 * np.log10(nmse_loss(H_hat, H_true).item() + 1e-10)

    for CR in args.cr:
        print(f"\n  Training CR=1/{CR}  (M={2*args.N_t*args.N_p//CR})  on: {DEVICE}")
        model = CsiNet(args.N_t, args.N_p, CR).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sch   = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)
        best, patience, ctr = float("inf"), 50, 0

        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); nmse_loss(model(xb), yb).backward(); opt.step()
            sch.step()
            if ep % 50 == 0 or ep == args.epochs:
                model.eval()
                vals = []
                with torch.no_grad():
                    for xb, yb in va_dl:
                        vals.append(nmse_db_t(model(xb.to(DEVICE)), yb.to(DEVICE)))
                v = float(np.mean(vals))
                print(f"    Epoch {ep:4d} | val NMSE {v:.2f} dB | lr {sch.get_last_lr()[0]:.1e}")
                if v < best:
                    best, ctr = v, 0
                    torch.save(model.state_dict(), f"{MODEL_DIR}/csinet_CR{CR}.pth")
                else:
                    ctr += 1
                    if ctr >= patience:
                        print(f"    Early stop at epoch {ep}"); break
        print(f"  ✓ Best val NMSE = {best:.2f} dB")
"""

"""

def run_train(args):
    print("=" * 60)
    print("STEP 2 — Training CsiNet (Path A)")
    print("=" * 60)
    H_tr = torch.from_numpy(np.load(f"{DATA_DIR}/H_train.npy"))
    H_va = torch.from_numpy(np.load(f"{DATA_DIR}/H_val.npy"))
    tr_dl = DataLoader(TensorDataset(H_tr, H_tr), batch_size=512,
                       shuffle=True,  num_workers=0, pin_memory=False)
    va_dl = DataLoader(TensorDataset(H_va, H_va), batch_size=512,
                       shuffle=False, num_workers=0, pin_memory=False)

    def nmse_db_t(H_hat, H_true):
        return 10 * np.log10(nmse_loss(H_hat, H_true).item() + 1e-10)

    for CR in args.cr:
        ckpt_path = f"{MODEL_DIR}/csinet_CR{CR}.pth"

        # ── Skip-if-exists check ────────────────────────────────────────────
        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"\n  [SKIP] CR=1/{CR} — checkpoint already exists at {ckpt_path}")
            print(f"         (pass --force_retrain to override and retrain anyway)")
            continue

        print(f"\n  Training CR=1/{CR}  (M={2*args.N_t*args.N_p//CR})  on: {DEVICE}")
        model = CsiNet(args.N_t, args.N_p, CR).to(DEVICE)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sch   = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-5)

        best      = float("inf")
        patience  = 50
        ctr       = 0
        min_delta = args.min_delta   # dB — improvement smaller than this = noise, not progress

        for ep in range(1, args.epochs + 1):
            model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); nmse_loss(model(xb), yb).backward(); opt.step()
            sch.step()

            if ep % 50 == 0 or ep == args.epochs:
                model.eval()
                vals = []
                with torch.no_grad():
                    for xb, yb in va_dl:
                        vals.append(nmse_db_t(model(xb.to(DEVICE)), yb.to(DEVICE)))
                v = float(np.mean(vals))

                # ── Smart early stopping logic ──────────────────────────────
                if v < best - min_delta:
                    # Genuine improvement — reset patience clock
                    improvement = best - v
                    best, ctr = v, 0
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"    Epoch {ep:4d} | val NMSE {v:.2f} dB | lr {sch.get_last_lr()[0]:.1e} "
                          f"| ✓ improved {improvement:.3f} dB")
                elif v < best:
                    # Tiny improvement — still save (it IS the best model so far)
                    # but don't reset patience, since it's not meaningful progress
                    best = v
                    torch.save(model.state_dict(), ckpt_path)
                    ctr += 1
                    print(f"    Epoch {ep:4d} | val NMSE {v:.2f} dB | lr {sch.get_last_lr()[0]:.1e} "
                          f"| (marginal, patience {ctr}/{patience})")
                    if ctr >= patience:
                        print(f"    Early stop at epoch {ep} (plateaued — only marginal gains for {patience} checks)")
                        break
                else:
                    ctr += 1
                    print(f"    Epoch {ep:4d} | val NMSE {v:.2f} dB | lr {sch.get_last_lr()[0]:.1e} "
                          f"| (no improvement, patience {ctr}/{patience})")
                    if ctr >= patience:
                        print(f"    Early stop at epoch {ep}")
                        break

        print(f"  ✓ Best val NMSE = {best:.2f} dB  → saved {ckpt_path}")
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — 3GPP TYPE II HIGH-RESOLUTION PMI CODEBOOK  (v4 fixed)
# ═══════════════════════════════════════════════════════════════════════════════

def build_type2_base_beams(N1=8, N2=2, O1=4, O2=2):
    """
    Generates the master 2D-DFT beam space for one polarisation.
    Returns (N1*O1 * N2*O2, N1*N2) = (128, 16) complex array.
    """
    beams = []
    for m in range(N2 * O2):
        for l in range(N1 * O1):
            u = np.exp(1j*2*np.pi*l*np.arange(N1)/(N1*O1)) / np.sqrt(N1)
            v = np.exp(1j*2*np.pi*m*np.arange(N2)/(N2*O2)) / np.sqrt(N2)
            b = np.kron(v, u)
            beams.append(b / np.linalg.norm(b))
    return np.array(beams)   # (128, 16)

_BASE_BEAMS = None
def get_base_beams():
    global _BASE_BEAMS
    if _BASE_BEAMS is None:
        _BASE_BEAMS = build_type2_base_beams()
    return _BASE_BEAMS

def pmi_type2_reconstruct_sf(H_sf_batch, N_t=32, L=4):
    """
    3GPP Type II PMI reconstruction in spatial-frequency domain.

    Key fixes vs v4:
      Fix A: H_subspace = W1 @ H  (not W1.conj() @ H)
      Fix B: Dual-pol handled correctly — beams applied to full N_t port vector
      Fix C: Reconstruction is W1^H @ W2 which equals sum of L rank-1 terms

    3GPP Type II structure:
      W1 : selects L beam directions from the DFT grid (wideband)
      W2 : per-subcarrier complex coefficients for each selected beam (subband)
      W  = W1 * W2  (linear combination)
      H_hat = W @ W^H @ H  (projection onto W subspace)
    """
    BB = get_base_beams()   # (128, 16) — one-pol DFT beams
    B, N_t_, N_c = H_sf_batch.shape
    n_per_pol = N_t // 2   # 16 for 32T

    H_hat = np.zeros_like(H_sf_batch, dtype=complex)

    for s in range(B):
        H = H_sf_batch[s]   # (N_t, N_c) = (32, 64)

        # ── W1: Wideband beam group selection ──────────────────────────────
        # Score each DFT beam against H by measuring captured power
        # Use both polarisations: H[:16,:] for +pol, H[16:,:] for -pol
        H_pol1 = H[:n_per_pol, :]   # (16, N_c)
        H_pol2 = H[n_per_pol:, :]   # (16, N_c)

        # Power captured by each beam b: ||b^H @ H_pol||^2 summed over subcarriers
        scores1 = np.sum(np.abs(BB @ H_pol1)**2, axis=1)   # (128,)
        scores2 = np.sum(np.abs(BB @ H_pol2)**2, axis=1)   # (128,)
        scores  = scores1 + scores2                          # combine both pols

        # Select top-L beam indices
        top_L = np.argsort(scores)[-L:]                     # (L,)
        W1_beams = BB[top_L]                                 # (L, 16)

        # Build full N_t x N_t beam matrix (dual-pol, co-phase φ=+1)
        # W1_full[b] applies same beam to both +pol and -pol blocks
        W1_full = np.zeros((L, N_t), dtype=complex)
        W1_full[:, :n_per_pol]  = W1_beams / np.sqrt(2)
        W1_full[:, n_per_pol:]  = W1_beams / np.sqrt(2)    # co-phase +1

        # ── W2: Subband linear combination coefficients ─────────────────────
        # Fix A: project H into beam subspace using W1 (not W1.conj())
        # H_sub[l, k] = energy of beam l on subcarrier k
        H_sub = W1_full @ H                                  # (L, N_c)

        # 3-bit amplitude + 3-bit phase quantisation (3GPP Type II constraint)
        amp   = np.abs(H_sub)
        phase = np.angle(H_sub)

        # Quantise phase to 8-PSK (3 bits: 8 levels)
        phase_q = np.round(phase / (2*np.pi/8)) * (2*np.pi/8)

        # Quantise amplitude to 3 bits (8 levels, normalised per subcarrier)
        amp_max = np.max(amp, axis=0, keepdims=True) + 1e-8
        amp_q   = np.round((amp / amp_max) * 7) / 7 * amp_max

        W2 = amp_q * np.exp(1j * phase_q)                   # (L, N_c)

        # ── Reconstruct H ───────────────────────────────────────────────────
        # H_hat = W1^H @ W2
        # = sum over L beams of: outer(w1_beam, w2_coefficients)
        # Each term is a rank-1 contribution — L terms summed = rank-L reconstruction
        H_hat[s] = W1_full.conj().T @ W2                    # (N_t, N_c)

    return H_hat


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METRICS  (Fix 8: all in spatial-frequency domain)
# ═══════════════════════════════════════════════════════════════════════════════

def angular_delay_to_sf(H_ad, N_c=64):
    """
    (B, N_t, N_p) complex angular-delay  →  (B, N_t, N_c) complex spatial-freq.
    Zero-pads the truncated delay dimension then applies inverse transforms.
    """
    B, N_t, N_p = H_ad.shape
    H_full = np.zeros((B, N_t, N_c), dtype=complex)
    H_full[:, :, :N_p] = H_ad
    H_sf = np.fft.ifft(H_full, axis=1)   # angular → spatial
    H_sf = np.fft.fft(H_sf,   axis=2)   # delay   → frequency
    return H_sf

def nmse_db_sf(H_hat_sf, H_true_sf):
    """NMSE in dB computed in spatial-frequency domain."""
    err = np.mean(np.abs(H_hat_sf - H_true_sf)**2, axis=(1,2))
    ref = np.mean(np.abs(H_true_sf)**2,             axis=(1,2))
    return float(10 * np.log10(np.mean(err / (ref + 1e-8)) + 1e-10))

def cosine_sim_sf(H_hat_sf, H_true_sf):
    """
    Spatial Geometry Cosine Similarity (SGCS) in spatial-frequency domain.
    This is the primary KPI from 3GPP TR 38.843 Table 1.
    Range [0, 1] — higher is better.
    """
    h  = H_hat_sf.reshape(len(H_hat_sf),   -1)
    ht = H_true_sf.reshape(len(H_true_sf), -1)
    n  = np.sum((h * ht.conj()).real, axis=1)
    d  = np.linalg.norm(h, axis=1) * np.linalg.norm(ht, axis=1) + 1e-8
    return float(np.mean(n / d))

def bf_gain_db_sf(H_true_sf, H_hat_sf, n_layers=1, max_samples=200):
    """
    Beamforming gain loss vs optimal SVD precoder, in spatial-frequency domain.
    Uses U[:,0] (left singular vector) as TX precoder — Fix 2 from v2.
    """
    ratios = []
    for i in range(min(len(H_true_sf), max_samples)):
        H  = H_true_sf[i]   # (N_t, N_c)
        Hh = H_hat_sf[i]
        try:
            U,  _, _ = np.linalg.svd(H,  full_matrices=False)
            Uh, _, _ = np.linalg.svd(Hh, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        W_opt = U[:,  :n_layers]           # (N_t, n_layers)
        W_rec = Uh[:, :n_layers]
        g_opt = np.linalg.norm(W_opt.conj().T @ H)**2
        g_rec = np.linalg.norm(W_rec.conj().T @ H)**2
        if g_opt > 1e-10:
            ratios.append(g_rec / g_opt)
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    return 10 * np.log10(mean_ratio + 1e-10)

def sgcs_gain_pct(rho_ai, rho_pmi):
    """
    SGCS gain over PMI in percent — what TR 38.843 Table 1 reports.
    Reported range: 1.4% to 21.4% across evaluators.
    """
    return (rho_ai - rho_pmi) / (1.0 - rho_pmi + 1e-8) * 100.0

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_scalers(scenarios):
    path = f"{DATA_DIR}/scalers.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        if d:
            return d
    print("  WARNING: scalers.json missing or empty — using scaler=1.0")
    return {sc: 1.0 for sc in scenarios}

def evaluate_both_paths(H_norm_test, H_sf_test, args,
                        scaler=1.0, H_sf_scaler=None,   # ← add H_sf_scaler
                        batch_size=500, label="All CDL"):
    """
    Evaluate AI (Path A) and PMI (Path B) on the same test set.

    Fix 8: BOTH paths are evaluated in spatial-frequency domain.
    ─────────────────────────────────────────────────────────────
    AI path:
      H_norm (angular-delay, normalised)
        → model forward pass → Ĥ_norm (angular-delay)
        → denormalise → angular_delay_to_sf → Ĥ_sf
        → compare with H_sf_test

    PMI path:
      H_sf_test (spatial-frequency)
        → pmi_reconstruct_sf → Ĥ_pmi_sf
        → compare with H_sf_test

    Both compared against the SAME H_sf_test ground truth.
    No truncation artefacts. Fair comparison.
    """
    if H_sf_scaler is None:
        H_sf_scaler = float(np.max(np.abs(H_sf_test)) + 1e-8)
    print(f"\n  Evaluating: {label}  ({len(H_norm_test)} samples)")
    CR_results = {}

    for CR in args.cr:
        ckpt = f"{MODEL_DIR}/csinet_CR{CR}.pth"
        if not os.path.exists(ckpt):
            print(f"    [SKIP] No checkpoint for CR=1/{CR}")
            continue

        model = CsiNet(args.N_t, args.N_p, CR).to(DEVICE)
        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()

        ai_nmse, ai_rho, ai_bf = [], [], []
        pm_nmse, pm_rho, pm_bf = [], [], []
        n_batches = (len(H_norm_test) + batch_size - 1) // batch_size

        for b in range(n_batches):
            sl      = slice(b * batch_size, (b+1) * batch_size)
            H_b     = H_norm_test[sl]      # (B,2,N_t,N_p) float32 normalised
            H_sf_b  = H_sf_test[sl]       # (B,N_t,N_c)   complex raw

            # Normalise H_sf to same scale as AI path (Bug 1+3 fix)
            # Both AI output and PMI reconstruction compared against this
            # REPLACE WITH:

            H_sf_scaled = H_sf_b / H_sf_scaler

            # ── Path A: AI autoencoder ────────────────────────────────────
            Ht = torch.from_numpy(H_b).to(DEVICE)
            with torch.no_grad():
                H_hat_norm = model(Ht).cpu().numpy()   # (B,2,N_t,N_p)

            # Convert AI output to spatial-frequency
            # Replace with:
            H_hat_ad = (H_hat_norm[:,0] + 1j*H_hat_norm[:,1]) * scaler  # restore AD scale
            H_hat_sf = angular_delay_to_sf(H_hat_ad, args.N_c)           # → sf domain

            # Then rescale to H_sf domain for fair comparison:
            H_hat_sf_scaled = H_hat_sf / H_sf_scaler

            # Bug 3 fix: compare against H_sf_scaled not H_sf_b
            ai_nmse.append(nmse_db_sf(H_hat_sf_scaled, H_sf_scaled))
            ai_rho.append( cosine_sim_sf(H_hat_sf_scaled, H_sf_scaled))
            if b < 2:
                ai_bf.append(bf_gain_db_sf(H_sf_scaled, H_hat_sf_scaled))

            # ── Path B: PMI codebook ──────────────────────────────────────
            # Run PMI on scaled H so reconstruction matches AI scale
            H_pmi_sf = pmi_type2_reconstruct_sf(H_sf_scaled, args.N_t, L=4)

            # Bug 2 fix: append ONCE only (remove the duplicate lines)
            pm_nmse.append(nmse_db_sf(H_pmi_sf, H_sf_scaled))
            pm_rho.append( cosine_sim_sf(H_pmi_sf, H_sf_scaled))
            if b < 2:
                pm_bf.append(bf_gain_db_sf(H_sf_scaled, H_pmi_sf))


        r = dict(
            ai_nmse = float(np.mean(ai_nmse)),
            ai_rho  = float(np.mean(ai_rho)),
            ai_bf   = float(np.nanmean(ai_bf)  if ai_bf  else float("nan")),
            pm_nmse = float(np.mean(pm_nmse)),
            pm_rho  = float(np.mean(pm_rho)),
            pm_bf   = float(np.nanmean(pm_bf)  if pm_bf  else float("nan")),
        )

        gain_db  = r["pm_nmse"] - r["ai_nmse"]
        gain_pct = sgcs_gain_pct(r["ai_rho"], r["pm_rho"])
        verdict  = "✓ AI WINS" if gain_db > 0 else "✗ AI LOSES"

        print(f"\n    CR=1/{CR}:")
        print(f"      AI:  NMSE={r['ai_nmse']:+.2f} dB | ρ={r['ai_rho']:.4f} | BF={r['ai_bf']:.2f} dB")
        print(f"      PMI: NMSE={r['pm_nmse']:+.2f} dB | ρ={r['pm_rho']:.4f} | BF={r['pm_bf']:.2f} dB")
        print(f"      NMSE gain:  {gain_db:+.2f} dB  {verdict}")
        print(f"      SGCS gain:  {gain_pct:+.1f}%   (TR 38.843 reports 1.4–21.4%)")

        CR_results[CR] = r
    return CR_results

def plot_comparison(all_results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "CsiNet (AI) vs Type II DFT Codebook (PMI Baseline)\n"
        "CDL-C / CDL-D — metrics in spatial-frequency domain — 3GPP TR 38.843",
        fontsize=13, fontweight="bold")

    colours = {"All CDL":"#185FA5","CDL-C":"#0F6E56","CDL-D":"#E65100"}
    m_ai    = {"All CDL":"o","CDL-C":"D","CDL-D":"v"}
    m_pmi   = {"All CDL":"X","CDL-C":"D","CDL-D":"v"}

    for label, results in all_results.items():
        if not results: continue
        CRs = sorted(results.keys())
        x   = [f"1/{cr}" for cr in CRs]
        col = colours.get(label, "#444")
        ls  = "-"  if label == "All CDL" else "--"
        lw  = 2.5  if label == "All CDL" else 1.8

        ai_n = [results[cr]["ai_nmse"] for cr in CRs]
        pm_n = [results[cr]["pm_nmse"] for cr in CRs]
        ai_r = [results[cr]["ai_rho"]  for cr in CRs]
        pm_r = [results[cr]["pm_rho"]  for cr in CRs]

        axes[0].plot(x, ai_n, ls,  color=col, lw=lw, marker=m_ai.get(label,"o"),
                     ms=8, label=f"AI {label}")
        axes[0].plot(x, pm_n, ":", color=col, lw=1.5, marker=m_pmi.get(label,"X"),
                     ms=7, alpha=0.8, label=f"PMI {label}")
        axes[1].plot(x, ai_r, ls,  color=col, lw=lw, marker=m_ai.get(label,"o"),
                     ms=8, label=f"AI {label}")
        axes[1].plot(x, pm_r, ":", color=col, lw=1.5, marker=m_pmi.get(label,"X"),
                     ms=7, alpha=0.8, label=f"PMI {label}")

        if label == "All CDL":
            ai_b = [results[cr]["ai_bf"] for cr in CRs]
            pm_b = [results[cr]["pm_bf"] for cr in CRs]
            axes[2].plot(x, ai_b, "o-",  color="#0F6E56", lw=2.5, ms=9, label="AI")
            axes[2].plot(x, pm_b, "X--", color="#993C1D", lw=2,   ms=9,
                         label="PMI (Type I DFT)")

    # Add SGCS gain annotation on middle plot
    for label, results in all_results.items():
        if label != "All CDL": continue
        CRs = sorted(results.keys())
        for cr in CRs:
            r    = results[cr]
            pct  = sgcs_gain_pct(r["ai_rho"], r["pm_rho"])
            xi   = CRs.index(cr)
            axes[1].annotate(f"+{pct:.0f}%",
                             xy=(xi, r["ai_rho"]),
                             xytext=(xi + 0.05, r["ai_rho"] + 0.04),
                             fontsize=8, color="#185FA5", fontweight="bold")

    axes[0].set_title("① NMSE (dB) — lower is better", fontweight="bold")
    axes[0].set_xlabel("Compression Ratio CR")
    axes[0].set_ylabel("NMSE (dB)  [spatial-frequency domain]")
    # axes[0].invert_yaxis()
    axes[0].axhline(0, color='#B4B2A9', lw=0.8, ls='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_facecolor("#FAFAF8")

    axes[1].set_title("② Cosine Similarity ρ (SGCS)\n3GPP TR 38.843 primary KPI — higher is better",
                      fontweight="bold")
    axes[1].set_xlabel("Compression Ratio CR")
    axes[1].set_ylabel("ρ  [spatial-frequency domain]")
    axes[1].set_ylim(0.0, 1.08)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_facecolor("#FAFAF8")

    axes[2].set_title("③ Beamforming Gain Loss (dB)\nvs perfect SVD precoder", fontweight="bold")
    axes[2].set_xlabel("Compression Ratio CR")
    axes[2].set_ylabel("BF Gain Loss (dB)")
    axes[2].axhline(-0.5, color="#EF9F27", ls="--", lw=1.5, label="Target: > −0.5 dB")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].set_facecolor("#FAFAF8")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nPlot saved: {out_path}")
    plt.close()

def run_evaluate(args):
    print("=" * 60)
    print("STEP 3 — Evaluating Both Paths (v3 — spatial-frequency metrics)")
    print("=" * 60)

    scenarios = args.cdl.split(",") if args.cdl != "all" else list(CDL_PARAMS.keys())
    scalers   = load_scalers(scenarios)
    all_results = {}

    # ── Combined test set ─────────────────────────────────────────────────────
    H_test   = np.load(f"{DATA_DIR}/H_test.npy")
    H_sf_all = np.load(f"{DATA_DIR}/H_sf_test_all.npy")
    # Fix 10: use mean scaler across scenarios for combined set
    mean_sc  = float(np.mean([scalers.get(sc, 1.0) for sc in scenarios]))
    print(f"  Mean scaler for combined set: {mean_sc:.4f}")

    H_sf_all_loaded = np.load(f"{DATA_DIR}/H_sf_test_all.npy")
    H_sf_scaler = float(np.max(np.abs(H_sf_all_loaded)) + 1e-8)
    print(f"  H_sf scaler: {H_sf_scaler:.4f}")

    all_results["All CDL"] = evaluate_both_paths(
    H_test, H_sf_all, args,
    scaler=mean_sc,           # keep existing — used for AI path
    H_sf_scaler=H_sf_scaler,  # NEW — used for PMI path
    label="All CDL")

    # ── Per-scenario ──────────────────────────────────────────────────────────
    # For per-scenario, compute separately:
    for sc in scenarios:
        np_path = f"{DATA_DIR}/H_test_CDL{sc}.npy"
        sf_path = f"{DATA_DIR}/H_sf_test_CDL{sc}.npy"
        if os.path.exists(np_path) and os.path.exists(sf_path):
            H_sf_sc = np.load(sf_path)
            H_sf_scaler_sc = float(np.max(np.abs(H_sf_sc)) + 1e-8)
            all_results[f"CDL-{sc}"] = evaluate_both_paths(
                np.load(np_path), H_sf_sc, args,
                scaler=scalers.get(sc, 1.0),
                H_sf_scaler=H_sf_scaler_sc,
                label=f"CDL-{sc}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(f"{MODEL_DIR}/eval_results_v3.json", "w") as f:
        json.dump({k: {str(cr): v for cr,v in res.items()}
                   for k,res in all_results.items()}, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(all_results, f"{MODEL_DIR}/comparison_plot_v5.png")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "═"*82)
    print(f"{'Label':<10} {'CR':>5} {'AI NMSE':>10} {'PMI NMSE':>10} "
          f"{'Gain dB':>9} {'AI ρ':>8} {'PMI ρ':>8} {'SGCS %':>8}")
    print("─"*82)
    for label, res in all_results.items():
        for cr in sorted(res.keys()):
            r        = res[cr]
            gain_db  = r["pm_nmse"] - r["ai_nmse"]
            gain_pct = sgcs_gain_pct(r["ai_rho"], r["pm_rho"])
            print(f"{label:<10} 1/{cr:<3} "
                  f"{r['ai_nmse']:>10.2f} {r['pm_nmse']:>10.2f} "
                  f"{gain_db:>+9.2f} {r['ai_rho']:>8.4f} "
                  f"{r['pm_rho']:>8.4f} {gain_pct:>+7.1f}%")
    print("═"*82)
    print("\nExpected ranges (COST2100 / CDL benchmark):")
    print("  PMI ρ  : 0.60 – 0.85  (Type I DFT codebook)")
    print("  AI  ρ  : 0.85 – 0.99  (CsiNet, depends on CR)")
    print("  SGCS % : +5%  – +25%  (your gain over PMI)")
    print("  TR 38.843 reported: 1.4% – 21.4% across evaluators")

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CSI Feedback AI vs PMI — v5.0")
    ap.add_argument("--phase",  choices=["generate","train","evaluate","all"],
                    default="all")
    ap.add_argument("--N_t",    type=int,   default=32)
    ap.add_argument("--N_c",    type=int,   default=64)
    ap.add_argument("--N_p",    type=int,   default=32)
    ap.add_argument("--cdl",    type=str,   default="C,D")
    ap.add_argument("--cr",     type=int,   nargs="+", default=[4,8,16,32])
    ap.add_argument("--epochs", type=int,   default=1000)
    ap.add_argument("--snr",    type=float, default=20)
    ap.add_argument("--force_retrain", action="store_true",
                    help="Retrain even if a checkpoint already exists")
    ap.add_argument("--min_delta", type=float, default=0.02,
                    help="Minimum NMSE improvement (dB) to reset patience counter")
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Config: N_t={args.N_t}, N_c={args.N_c}, N_p={args.N_p}, "
          f"CDL={args.cdl}, CR={args.cr}, epochs={args.epochs}")

    if args.phase in ("generate", "all"): run_generate(args)
    if args.phase in ("train",    "all"): run_train(args)
    if args.phase in ("evaluate", "all"): run_evaluate(args)