"""
Fixes the H_sf_test construction in load_cost2100() to recentre
COST2100's [0,1]-normalised real/imag channels to [-1,1] before
treating them as complex values for the PMI codebook path.

Without this fix, every sample carries a spurious +0.5+0.5j DC bias,
which corrupts angular_delay_to_sf()'s FFT-based reconstruction used
for the Type II PMI baseline (though NOT the AI path, which is trained
directly on the [0,1] representation via sigmoid activations and needs
no change).

Usage:  python patch_h_sf_recenter.py
"""
import re

PATH = "csi_ai_v6.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

old = '''    H_ad_complex = H_test[:, 0] + 1j * H_test[:, 1]     # (N, 32, 32) complex
    H_sf_test    = angular_delay_to_sf(H_ad_complex, N_c=N_c_full)'''

new = '''    # COST2100 real/imag channels are affine-normalised to [0,1] by the
    # dataset authors (CsiNet/CRNet convention). Recentre to [-1,1] before
    # treating as a signed complex signal -- otherwise every sample carries
    # a spurious +0.5+0.5j DC bias that corrupts the FFT-based
    # spatial-frequency reconstruction used by the Type II PMI codebook.
    # (The AI path needs no such correction -- CsiNet is trained directly
    # on the [0,1] representation via its sigmoid encoder/decoder output.)
    H_test_centered = 2.0 * H_test - 1.0                      # [0,1] -> [-1,1]
    H_ad_complex = H_test_centered[:, 0] + 1j * H_test_centered[:, 1]
    H_sf_test    = angular_delay_to_sf(H_ad_complex, N_c=N_c_full)'''

if old in src:
    src_new = src.replace(old, new, 1)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src_new)
    print("✓ Patched H_sf_test construction (added [0,1]→[-1,1] recentering).")
else:
    print("✗ Exact match not found -- pattern may differ slightly.")
    print("  Searching for a looser match...")
    loose_pattern = re.compile(
        r'H_ad_complex = H_test\[:, 0\] \+ 1j \* H_test\[:, 1\].*?\n\s*H_sf_test\s*=\s*angular_delay_to_sf\(H_ad_complex, N_c=N_c_full\)',
        re.DOTALL
    )
    if loose_pattern.search(src):
        src_new = loose_pattern.sub(new.strip(), src, count=1)
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(src_new)
        print("✓ Patched via loose pattern match.")
    else:
        print("✗ Could not find pattern at all. Manual edit needed.")
        print("  Find the line starting with 'H_ad_complex = H_test[:, 0]'")
        print("  and replace the two lines with the 'new' block above.")