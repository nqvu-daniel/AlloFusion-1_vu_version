#!/usr/bin/env python3
"""
Prefetch the ProtT5 model weights to a local directory using huggingface_hub.

This avoids large downloads during the first run by caching the checkpoint
up-front and optionally pointing PROT_T5_PATH at the local copy.

Usage:
  python scripts/prefetch_prot_t5.py \
      --repo Rostlab/prot_t5_xl_uniref50 \
      --dest models/prot_t5_xl_uniref50

Environment:
  - Set HUGGINGFACE_HUB_CACHE to control the global HF cache if desired
"""
import argparse
import os
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser(description="Prefetch ProtT5 weights locally")
    ap.add_argument("--repo", default="Rostlab/prot_t5_xl_uniref50",
                    help="Hugging Face repo id or local path")
    ap.add_argument("--dest", default="models/prot_t5_xl_uniref50",
                    help="Destination directory to store the snapshot")
    ap.add_argument("--token", default=None, help="Hugging Face token, if needed")
    ap.add_argument("--full", action="store_true",
                    help="Download the full repo snapshot. By default we fetch only the minimal files needed (no 600k/723k weights).")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    # Download a local snapshot; no symlinks for portability.
    # By default, restrict to the minimal files required by Transformers for T5Encoder:
    #  - config.json, tokenizer_config.json, special_tokens_map.json, spiece.model, pytorch_model.bin
    allow_patterns = None
    ignore_patterns = ["*.msgpack", "*.h5"]
    if not args.full:
        allow_patterns = [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
            "pytorch_model.bin",  # only the main weights file
        ]

    print(f"[prefetch_prot_t5] Downloading from {args.repo} to {args.dest}")
    if not args.full:
        print("[prefetch_prot_t5] Minimal mode: downloading only essential files (~3GB)")
        print("[prefetch_prot_t5] This avoids unnecessary 600k/723k checkpoint files")
    else:
        print("[prefetch_prot_t5] Full mode: downloading entire repository")

    local_path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.dest,
        local_dir_use_symlinks=False,
        token=args.token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    # Validate essential files
    essential_files = [
        "config.json",
        "tokenizer_config.json",
        "spiece.model",
        "pytorch_model.bin",
    ]
    missing = []
    for fname in essential_files:
        fpath = os.path.join(args.dest, fname)
        if not os.path.exists(fpath):
            missing.append(fname)

    if missing:
        print(f"[prefetch_prot_t5] ERROR: Missing essential files: {', '.join(missing)}", flush=True)
        exit(1)

    # Report success and size
    import subprocess
    try:
        result = subprocess.run(
            ["du", "-sh", args.dest],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"[prefetch_prot_t5] Download complete: {size} in {args.dest}")
        else:
            print(f"[prefetch_prot_t5] Download complete: {args.dest}")
    except Exception:
        print(f"[prefetch_prot_t5] Download complete: {args.dest}")

    print(f"[prefetch_prot_t5] All essential files validated. Set PROT_T5_PATH={args.dest}")


if __name__ == "__main__":
    main()
