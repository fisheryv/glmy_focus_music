"""Create and resumably upload the combined audited Hugging Face dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = ROOT / "hf_release"
REPO_NAME = "open-focus-classical-600"
RELEASE_FOLDER = RELEASE_ROOT / REPO_NAME


def read_token() -> str:
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    env_path = ROOT / ".env"
    if not env_path.is_file():
        raise RuntimeError("Missing .env and HUGGINGFACE_API_TOKEN environment variable")
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HUGGINGFACE_API_TOKEN":
            token = value.strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError("HUGGINGFACE_API_TOKEN is missing or empty")


def validate_release(folder: Path) -> None:
    required = [
        folder / "README.md",
        folder / "metadata" / "tracks.csv",
        folder / "metadata" / "licenses.csv",
        folder / "SHA256SUMS",
    ]
    audio = [
        path
        for path in (folder / "data").glob("**/*")
        if path.is_file() and path.name != "metadata.csv"
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing required files: {missing}")
    if len(audio) != 600:
        raise RuntimeError(f"Expected 600 audio files, found {len(audio)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--namespace",
        help="Hugging Face user or organization; defaults to the authenticated user",
    )
    parser.add_argument("--repo-name", default=REPO_NAME)
    args = parser.parse_args()

    token = read_token()
    api = HfApi(token=token)
    identity = api.whoami()
    namespace = args.namespace or identity["name"]
    print(f"Authenticated Hugging Face account: {identity['name']}")
    print(f"Target namespace: {namespace}")

    validate_release(RELEASE_FOLDER)
    repo_id = f"{namespace}/{args.repo_name}"
    url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not info.private:
        raise RuntimeError(f"Refusing to upload because {repo_id} is not private")
    print(f"Uploading {RELEASE_FOLDER} -> {url}")
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=RELEASE_FOLDER,
        repo_type="dataset",
        print_report=True,
        print_report_every=60,
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
