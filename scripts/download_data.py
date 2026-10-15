"""Download the text2bash seed/training data files from the Hugging Face
dataset repo they were published to with scripts/upload_dataset.py, restoring
them at the same paths they used to live at in git.

Usage:
    python scripts/download_data.py <repo_id>
    python scripts/download_data.py yourname/text2bash-data
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from common.data_files import DATA_FILES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "repo_id", help="HF dataset repo to download from, e.g. yourname/text2bash-data"
    )
    args = ap.parse_args()

    for rel_path in DATA_FILES:
        local_path = PROJECT_ROOT / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=rel_path)
        local_path.write_bytes(Path(cached_path).read_bytes())
        print(f"restored {rel_path}")

    print("Done.")


if __name__ == "__main__":
    main()
