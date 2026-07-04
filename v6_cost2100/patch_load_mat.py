"""
Run this once to patch load_mat() in csi_ai_v6.py correctly.
Usage:  python patch_load_mat.py
"""
import re

PATH = "csi_ai_v6.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

# Match the load_mat function body regardless of exact whitespace/comments
pattern = re.compile(
    r'def load_mat\(fname\):.*?return H\.transpose\(0, 3, 1, 2\)\.astype\("float32"\).*?\n',
    re.DOTALL
)

new_func = '''def load_mat(fname):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\\n"
                f"Copy DATA_Htrainin.mat / DATA_Hvalin.mat / DATA_Htestin.mat "
                f"into {DATA_DIR} first.")
        mat = scipy.io.loadmat(path)
        H   = mat["HT"].astype("float32")
        # COST2100 .mat files store H flattened as (N, 2048).
        # Original CsiNet/CRNet convention reshapes DIRECTLY to
        # (N, 2, 32, 32) -- channels-first -- no separate transpose step.
        # (2048 = 2 * 32 * 32: real/imag interleaved with N_t, N_p in
        # MATLAB's native flatten order, which numpy's default C-order
        # reshape reproduces correctly here.)
        N = H.shape[0]
        H = H.reshape(N, 2, 32, 32)
        return H
'''

if pattern.search(src):
    src_new = pattern.sub(new_func, src, count=1)
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src_new)
    print("✓ Patched load_mat() successfully.")
else:
    print("✗ Could not find load_mat() pattern to replace.")
    print("  Manual edit needed -- see instructions below.")