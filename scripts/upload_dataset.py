"""Upload the large text2bash seed/training data files to a Hugging Face
dataset repo, so they can be removed from git and restored on demand with
scripts/download_data.py.

NOT run automatically by anything else in this repo -- run it once, by hand,
after reviewing the file list in common/data_files.py. Requires the HF_TOKEN
environment variable (a Hugging Face write token).

Usage:
    HF_TOKEN=hf_... python scripts/upload_dataset.py <repo_id>
    HF_TOKEN=hf_... python scripts/upload_dataset.py yourname/text2bash-data
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from common.data_files import DATA_FILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo_id", help="Target HF dataset repo, e.g. yourname/text2bash-data")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "error: set the HF_TOKEN environment variable to a Hugging Face write token",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    for rel_path in DATA_FILES:
        local_path = PROJECT_ROOT / rel_path
        if not local_path.exists():
            print(f"skip (not found): {rel_path}", file=sys.stderr)
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"uploaded {rel_path}")

    print(f"Done. Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
