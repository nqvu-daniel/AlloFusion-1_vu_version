#!/usr/bin/env python3
"""
Fetch assets (model weights / datasets) from a GitHub repository's Releases.

Examples
- Download AlloFusion CNN weights to myModel/trial1.h5 (repo root):
    python scripts/fetch_release_assets.py \
        --repo hjb-001/AlloFusion \
        --pattern trial1.h5 \
        --dest myModel

- Download training datasets to features_data/diversity:
    python scripts/fetch_release_assets.py \
        --repo hjb-001/AlloFusion \
        --pattern train_dataset_*.pkl \
        --dest features_data/diversity

Notes
- Requires internet access.
- Uses the GitHub API to find the latest (or chosen) release and downloads assets
  whose names match the provided glob pattern.
"""
import argparse
import fnmatch
import json
import os
import sys
import urllib.request


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "AlloFusion-fetch/1.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def list_assets(repo: str, tag: str | None) -> list[dict]:
    if tag:
        url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        data = json.loads(_get(url))
        assets = data.get("assets", [])
    else:
        url = f"https://api.github.com/repos/{repo}/releases/latest"
        data = json.loads(_get(url))
        assets = data.get("assets", [])
    return assets


def download_asset(asset: dict, dest_dir: str) -> str:
    name = asset.get("name")
    url = asset.get("browser_download_url")
    if not name or not url:
        raise RuntimeError("Malformed asset in release JSON")
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, name)
    print(f"[fetch] Downloading {name} -> {out_path}")
    req = urllib.request.Request(url, headers={"User-Agent": "AlloFusion-fetch/1.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Fetch assets from GitHub Releases")
    ap.add_argument("--repo", required=True, help="GitHub repo, e.g. hjb-001/AlloFusion")
    ap.add_argument("--pattern", required=True, help="Glob pattern to match asset names (e.g., trial1.h5 or train_dataset_*.pkl)")
    ap.add_argument("--dest", required=True, help="Destination directory")
    ap.add_argument("--tag", default=None, help="Release tag (default: latest)")
    ap.add_argument("--extract", action="store_true", help="If set, auto-extract .zip/.tar(.gz) archives into dest")
    args = ap.parse_args()

    try:
        assets = list_assets(args.repo, args.tag)
    except Exception as e:
        print(f"Error: failed to query GitHub API for {args.repo}: {e}", file=sys.stderr)
        sys.exit(1)

    matched = [a for a in assets if fnmatch.fnmatch(a.get("name", ""), args.pattern)]
    if not matched:
        print(f"No assets matched pattern '{args.pattern}' in releases of {args.repo}", file=sys.stderr)
        print("Available assets:")
        for a in assets:
            print(" -", a.get("name"))
        sys.exit(2)

    extracted = []
    for a in matched:
        try:
            out = download_asset(a, args.dest)
            print(f"[OK] Saved {out}")
            if args.extract:
                lower = out.lower()
                if lower.endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(out, 'r') as zf:
                        zf.extractall(args.dest)
                        extracted.append(out)
                        print(f"[OK] Extracted {out} -> {args.dest}")
                elif lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz'):
                    import tarfile
                    mode = 'r:gz' if lower.endswith('.gz') or lower.endswith('.tgz') else 'r:'
                    with tarfile.open(out, mode) as tf:
                        tf.extractall(args.dest)
                        extracted.append(out)
                        print(f"[OK] Extracted {out} -> {args.dest}")
        except Exception as e:
            print(f"Error downloading {a.get('name')}: {e}", file=sys.stderr)
            sys.exit(3)

    if extracted:
        print("\nNote: extracted archives; original files remain in dest.")
    print("\nDone. If you downloaded 'trial1.h5', ensure it is at 'myModel/trial1.h5' relative to your run directory. (Legacy 'all.h5' also supported.)")


if __name__ == "__main__":
    main()
