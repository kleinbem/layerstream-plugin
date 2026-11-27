#!/usr/bin/env python3
"""Rebuild the normalized Hugging Face dataset (Parquet/JSON/JSONL) from
benchmarks/results/ and sync both the HF Dataset and interactive HF Space.

Usage:
    # Preview what would be rebuilt & synced:
    python3 benchmarks/sync_hf.py --dry-run

    # Rebuild and publish dataset + interactive Space:
    python3 benchmarks/sync_hf.py \
        --dataset-repo kleinbem/layerstream-orin-nano-benchmarks \
        --space-repo kleinbem/layerstream-benchmarks
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPRO_SCRIPTS = (
    "bench_llama_cpp.py",
    "bench_accelerate.py",
    "bench_airllm.py",
    "bench_colibri.py",
    "export_hf_dataset.py",
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Rebuild & sync LayerStream HF Dataset + Space")
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "benchmarks/results",
    )
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=repo_root / "benchmarks/hf-dataset",
    )
    ap.add_argument(
        "--space-dir",
        type=Path,
        default=repo_root / "benchmarks/hf-space",
    )
    ap.add_argument(
        "--dataset-repo",
        default="kleinbem/layerstream-orin-nano-benchmarks",
    )
    ap.add_argument(
        "--space-repo",
        default="kleinbem/layerstream-benchmarks",
    )
    ap.add_argument("--skip-space", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--message", default="Sync updated LayerStream multi-hardware benchmarks")
    args = ap.parse_args()

    export_script = repo_root / "benchmarks/export_hf_dataset.py"
    subprocess.run(  # noqa: S603 — explicit argv list invoking repo-local export script
        [
            sys.executable,
            str(export_script),
            "--results-dir",
            str(args.results_dir),
            "--out-dir",
            str(args.dataset_dir),
        ],
        check=True,
    )

    if args.dry_run:
        print(f"[dry-run] Would upload {args.dataset_dir} + scripts/ -> {args.dataset_repo}")
        if not args.skip_space:
            print(f"[dry-run] Would upload {args.space_dir} + benchmarks.json -> {args.space_repo}")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        stage_ds = Path(tmp) / "dataset"
        shutil.copytree(args.dataset_dir, stage_ds)
        scripts_dir = stage_ds / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        for name in REPRO_SCRIPTS:
            src = repo_root / "benchmarks" / name
            if src.is_file():
                shutil.copy2(src, scripts_dir / name)

        api.create_repo(
            repo_id=args.dataset_repo, repo_type="dataset", private=False, exist_ok=True
        )
        api.upload_folder(
            folder_path=str(stage_ds),
            repo_id=args.dataset_repo,
            repo_type="dataset",
            commit_message=args.message,
        )
        print(f"Synced dataset -> https://huggingface.co/datasets/{args.dataset_repo}")

        if not args.skip_space:
            stage_sp = Path(tmp) / "space"
            shutil.copytree(args.space_dir, stage_sp)
            shutil.copy2(args.dataset_dir / "data/benchmarks.json", stage_sp / "benchmarks.json")
            api.create_repo(
                repo_id=args.space_repo,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(stage_sp),
                repo_id=args.space_repo,
                repo_type="space",
                delete_patterns=["index.html"],
                commit_message=args.message,
            )
            print(f"Synced Space -> https://huggingface.co/spaces/{args.space_repo}")


if __name__ == "__main__":
    main()
