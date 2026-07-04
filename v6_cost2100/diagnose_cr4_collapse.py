"""
diagnose_cr4_collapse.py
─────────────────────────────────────────────────────────────────────────────
Standalone diagnostic: trains CR=4 in isolation for 20 epochs, printing
per-epoch loss, gradient norms (encoder vs decoder), and NaN/Inf checks
after every single optimizer step for the first 5 epochs, then every
epoch after that.

Does NOT touch your main csi_ai_v6.py or its checkpoints. Run from the
same folder (v6_cost2100) so it can import CsiNet, DATA_DIR, etc.

Usage:  python diagnose_cr4_collapse.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from csi_ai_v6 import CsiNet, nmse_loss, DATA_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

H_tr = torch.from_numpy(np.load(f"{DATA_DIR}/H_train.npy"))
H_va = torch.from_numpy(np.load(f"{DATA_DIR}/H_val.npy"))
print(f"H_train shape: {H_tr.shape}, H_val shape: {H_va.shape}")
print(f"H_train range: [{H_tr.min():.4f}, {H_tr.max():.4f}]  mean: {H_tr.mean():.4f}")

tr_dl = DataLoader(TensorDataset(H_tr, H_tr), batch_size=512,
                   shuffle=True, num_workers=0, pin_memory=False)

CR = 4
N_t, N_p = 32, 32
model = CsiNet(N_t, N_p, CR).to(DEVICE)

# Check initial state BEFORE any training
print("\n=== INITIAL WEIGHTS (before training) ===")
for name, p in model.named_parameters():
    if 'enc' in name and 'weight' in name:
        print(f"  {name:30s} std={p.std().item():.6f}  (should be nonzero)")

opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print("\n=== TRAINING WITH PER-STEP DIAGNOSTICS ===")
for ep in range(1, 21):
    model.train()
    epoch_losses = []
    for step, (xb, yb) in enumerate(tr_dl):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()

        pred = model(xb)
        loss = nmse_loss(pred, yb)

        # NaN/Inf check BEFORE backward
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  !!! Epoch {ep} Step {step}: loss is {loss.item()} (NaN/Inf) !!!")
            print(f"      pred min/max: {pred.min().item()}/{pred.max().item()}")
            print(f"      xb   min/max: {xb.min().item()}/{xb.max().item()}")
            sys.exit(1)

        loss.backward()

        # Check gradient norms BEFORE step (only first 5 epochs, first batch)
        if ep <= 5 and step == 0:
            enc_grad = sum(p.grad.abs().sum().item() for p in model.enc.parameters() if p.grad is not None)
            enc_fc_grad = model.enc_fc.weight.grad.abs().sum().item() if model.enc_fc.weight.grad is not None else -1
            dec_grad = sum(p.grad.abs().sum().item() for p in model.dec_fc.parameters() if p.grad is not None)
            print(f"  Epoch {ep} Step {step}: loss={loss.item():.6f}  "
                  f"enc_grad_sum={enc_grad:.6e}  enc_fc_grad_sum={enc_fc_grad:.6e}  dec_fc_grad_sum={dec_grad:.6e}")

        # Check for NaN gradients
        has_nan_grad = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        if has_nan_grad:
            print(f"  !!! Epoch {ep} Step {step}: NaN detected in gradients !!!")
            sys.exit(1)

        opt.step()
        epoch_losses.append(loss.item())

    mean_loss = np.mean(epoch_losses)
    nmse_db = 10 * np.log10(mean_loss + 1e-10)

    # Check encoder weight std after this epoch
    enc_std = model.enc[0].b[0].weight.std().item()
    bn_gamma_std = model.enc[0].b[1].weight.std().item()
    bn_gamma_mean = model.enc[0].b[1].weight.mean().item()

    print(f"Epoch {ep:3d} | train NMSE {nmse_db:7.2f} dB | "
          f"conv weight std={enc_std:.6f} | BN gamma mean={bn_gamma_mean:.4f} std={bn_gamma_std:.6f}")

    if enc_std < 1e-6:
        print(f"  !!! Encoder conv weights collapsed to ~zero at epoch {ep} !!!")
        break

print("\nDone. If BN gamma mean/std dropped toward 0 before the encoder")
print("weights collapsed, that confirms the BatchNorm-gamma-collapse hypothesis.")