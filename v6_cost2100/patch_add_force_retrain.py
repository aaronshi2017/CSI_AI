"""
Adds --force_retrain and --min_delta CLI arguments to csi_ai_v6.py,
and wires the skip-if-checkpoint-exists + smart-early-stopping logic
into run_train() if not already present.

Usage:  python patch_add_force_retrain.py
"""

PATH = "csi_ai_v6.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

changed = False

# ── 1. Add CLI arguments ────────────────────────────────────────────────────
old_arg = '''    ap.add_argument("--snr",    type=float, default=20)'''
new_arg = '''    ap.add_argument("--snr",    type=float, default=20)
    ap.add_argument("--force_retrain", action="store_true",
                    help="Retrain even if a checkpoint already exists")
    ap.add_argument("--min_delta", type=float, default=0.02,
                    help="Minimum NMSE improvement (dB) to reset patience counter")'''

if "--force_retrain" not in src:
    if old_arg in src:
        src = src.replace(old_arg, new_arg, 1)
        changed = True
        print("✓ Added --force_retrain and --min_delta CLI arguments.")
    else:
        print("✗ Could not find '--snr' argparse line to anchor the new arguments.")
else:
    print("  --force_retrain already present, skipping CLI arg addition.")

# ── 2. Add skip-if-exists check inside run_train's CR loop ──────────────────
old_loop_start = '''    for CR in args.cr:
        print(f"\\n  Training CR=1/{CR}  (M={2*args.N_t*args.N_p//CR})  on: {DEVICE}")
        model = CsiNet(args.N_t, args.N_p, CR).to(DEVICE)'''

new_loop_start = '''    for CR in args.cr:
        ckpt_path = f"{MODEL_DIR}/csinet_CR{CR}.pth"
        if os.path.exists(ckpt_path) and not args.force_retrain:
            print(f"\\n  [SKIP] CR=1/{CR} — checkpoint already exists at {ckpt_path}")
            print(f"         (pass --force_retrain to override and retrain anyway)")
            continue

        print(f"\\n  Training CR=1/{CR}  (M={2*args.N_t*args.N_p//CR})  on: {DEVICE}")
        model = CsiNet(args.N_t, args.N_p, CR).to(DEVICE)'''

if "ckpt_path = f\"{MODEL_DIR}/csinet_CR{CR}.pth\"" not in src:
    if old_loop_start in src:
        src = src.replace(old_loop_start, new_loop_start, 1)
        changed = True
        print("✓ Added skip-if-checkpoint-exists logic to run_train().")
    else:
        print("✗ Could not find run_train's CR loop start to patch skip-if-exists logic.")
        print("  (This is non-critical for your immediate need -- --force_retrain")
        print("   flag alone will let you retrain CR=4 once the CLI arg exists.)")
else:
    print("  Skip-if-exists logic already present, skipping.")

if changed:
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src)
    print("\\nFile saved.")
else:
    print("\\nNo changes made.")