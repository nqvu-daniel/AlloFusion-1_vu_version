import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ProtT5 model locally to avoid runtime downloads.")
    ap.add_argument("--repo", default="Rostlab/prot_t5_xl_uniref50", help="Hugging Face repo id")
    ap.add_argument("--dest", required=True, help="Destination directory (will be created)")
    ap.add_argument("--revision", default=None, help="Optional HF revision/tag/commit")
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install via `pip install huggingface-hub` "
            "(it is typically included with transformers)."
        ) from e

    print(f"[prefetch] Downloading {args.repo} -> {dest}")
    snapshot_download(
        repo_id=args.repo,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        revision=args.revision,
        # Keep only common model files; include tokenizer/model/config.
        allow_patterns=[
            "*.json",
            "*.bin",
            "*.model",
            "*.txt",
            "*.py",
        ],
    )
    print("[prefetch] Done.")
    print(f"[NEXT] export PROT_T5_PATH={dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

