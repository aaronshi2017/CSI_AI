"""
Fixes the AI-path DC-bias bug in evaluate_both_paths().

CsiNet's sigmoid output (H_hat_norm) lives in [0,1] -- same domain as the
raw COST2100 data -- but was being treated as already-signed when built
into a complex signal for spatial-frequency conversion. This carries the
same +0.5+0.5j DC-bias problem that was already fixed on the ground-truth
side (H_test -> H_sf_test), but was missed on the AI-reconstruction side.

After zero-padding N_p=32 -> N_c=1024, that DC bias completely dominates
the real (32-tap) signal, producing near-zero cosine similarity and wildly
positive NMSE for the AI path -- exactly the symptom observed.

Usage:  python patch_ai_recenter.py
"""

PATH = "csi_ai_v6.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

old = '''            H_hat_ad = (H_hat_norm[:,0] + 1j*H_hat_norm[:,1]) * scaler  # restore AD scale'''

new = '''            # Recentre [0,1] sigmoid output to [-1,1] before treating as a
            # signed complex signal -- same fix as applied to H_test when
            # building H_sf_test. Without this, the AI reconstruction
            # carries a spurious +0.5+0.5j DC bias that dominates after
            # zero-padding N_p=32 -> N_c=1024, corrupting rho and NMSE.
            H_hat_norm_centered = 2.0 * H_hat_norm - 1.0
            H_hat_ad = (H_hat_norm_centered[:,0] + 1j*H_hat_norm_centered[:,1]) * scaler'''

if old in src:
    src_new = src.replace(old, new, 1)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src_new)
    print("✓ Patched AI-path recentering in evaluate_both_paths().")
else:
    print("✗ Exact match not found.")
    print("  Find the line: H_hat_ad = (H_hat_norm[:,0] + 1j*H_hat_norm[:,1]) * scaler")
    print("  and replace with the recentered version (see 'new' block in this script).")