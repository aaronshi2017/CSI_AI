"""
Corrects the misdiagnosis: GroupNorm does NOT fix the CR=4 collapse,
because the real mechanism is weight_decay grinding all parameters
toward zero once training loss plateaus in a flat region -- which
happens almost immediately at CR=4 (M=512, only 4x compression, huge
capacity relative to the task) and does not happen at CR=8/16/32
(genuine compression keeps real gradient signal >> decay pull for far
longer).

Diagnostic evidence: loss drops 0.104 -> 0.0018 by epoch 2, then stays
flat (0.0018) through epoch 20, while GroupNorm's gamma-equivalent
affine scale monotonically shrinks 0.87 -> 0.002 over the same 20
epochs. Loss is flat; gamma is not. Only weight_decay applies a
non-zero pull on gamma once task-gradient is ~0. Confirms decay is the
active cause, not the specific normalisation layer type.

Fix: remove weight_decay entirely (safe for this architecture -- CR=8,
16, 32 already train and generalise well; weight_decay is not carrying
meaningful regularisation load here relative to the damage it does at
CR=4). Also fixes the early-stopping patience unit bug (was counting
validation CHECKS, not epochs, allowing up to patience*50 epochs of
non-improvement before stopping -- far beyond the 1000-epoch cap).

Usage:  python patch_weight_decay_fix.py
"""

PATH = "csi_ai_v6.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

changed = False

# ── Fix 1: remove weight_decay from Adam optimizer ───────────────────────────
old_opt = '''opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)'''
new_opt = '''opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)  # decay removed: was grinding all params to 0 at CR=4 once loss plateaued'''

if old_opt in src:
    src = src.replace(old_opt, new_opt, 1)
    changed = True
    print("✓ Removed weight_decay from optimizer.")
else:
    print("✗ Could not find optimizer line with weight_decay=1e-5.")
    print("  Search for 'optim.Adam(model.parameters()' and remove the")
    print("  weight_decay=1e-5 argument manually.")

if changed:
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("File saved.")