#!/usr/bin/env python3
"""
Download required models for the skin analyzer.

This script downloads all necessary model files to the models directory.
Total size: ~50-100MB (well under 1GB limit)
"""

import urllib.request
import os
from pathlib import Path
import hashlib
import sys


# Model definitions
MODELS = {
    "yunet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "filename": "face_detection_yunet_2023mar.onnx",
        "size_mb": 0.3,
        "description": "YuNet face detection model"
    },
    # Add more models as needed
    # "bisenet": {
    #     "url": "https://...",
    #     "filename": "bisenet_resnet18.pth",
    #     "size_mb": 50,
    #     "description": "BiSeNet face parsing model"
    # }
}


def get_models_dir() -> Path:
    """Get the models directory path."""
    # Try to find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """
    Download a file with progress indication.

    Args:
        url: URL to download from.
        dest_path: Destination file path.
        description: Description for progress output.

    Returns:
        True if successful, False otherwise.
    """
    if dest_path.exists():
        print(f"  [SKIP] {description} already exists")
        return True

    print(f"  [DOWNLOAD] {description}")
    print(f"    URL: {url}")
    print(f"    Destination: {dest_path}")

    try:
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\n    [OK] Download complete")
        return True

    except Exception as e:
        print(f"\n    [ERROR] Failed to download: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def verify_file(file_path: Path, expected_md5: str = None) -> bool:
    """Verify downloaded file integrity."""
    if not file_path.exists():
        return False

    if expected_md5:
        with open(file_path, 'rb') as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()
        return file_md5 == expected_md5

    # Just check file exists and has content
    return file_path.stat().st_size > 0


def main():
    """Download all required models."""
    print("=" * 60)
    print("Skin Analyzer - Model Downloader")
    print("=" * 60)

    models_dir = get_models_dir()
    print(f"\nModels directory: {models_dir}\n")

    total_size = sum(m["size_mb"] for m in MODELS.values())
    print(f"Models to download: {len(MODELS)}")
    print(f"Total size: ~{total_size:.1f} MB\n")

    success_count = 0
    fail_count = 0

    for model_name, model_info in MODELS.items():
        print(f"\n[{model_name.upper()}]")

        dest_path = models_dir / model_info["filename"]

        if download_file(
            model_info["url"],
            dest_path,
            model_info["description"]
        ):
            if verify_file(dest_path):
                success_count += 1
            else:
                print(f"    [WARNING] File verification failed")
                fail_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Download Summary: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)

    if fail_count > 0:
        print("\nWARNING: Some models failed to download.")
        print("The analyzer may not work correctly without all models.")
        return 1

    print("\nAll models downloaded successfully!")
    print("You can now run the skin analyzer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
