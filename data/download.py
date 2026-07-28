"""Download and extract the Crawford/CAT_DATASET cat facial-keypoint dataset.

Source: archive.org mirror of the original CAT_DATASET (Zhang, Sun & Tang),
also referenced on Kaggle as crawford/cat-dataset. Verified live as of
2026-07-28 — see CATalyze_ClaudeCode_Implementation_Brief.md Section 2.

Each image ``NNNN_MMM.jpg`` has a sibling ``NNNN_MMM.jpg.cat`` annotation file:
a single line ``"9 x0 y0 x1 y1 ... x8 y8"`` where the 9 points are, in order:
left eye, right eye, mouth, then two ears each described by 3 points
(base, tip, base) — see README for the exact left/right-ear point layout.
"""
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402

DATASET_URLS = [
    "https://archive.org/download/CAT_DATASET/CAT_DATASET_01.zip",
    "https://archive.org/download/CAT_DATASET/CAT_DATASET_02.zip",
]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"  extracting {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def consolidate(raw_dir: Path, images_dir: Path, annotations_dir: Path) -> int:
    """CAT_DATASET ships as CAT_00..CAT_06 folders each holding .jpg + .jpg.cat
    pairs. Flatten into images_dir / annotations_dir for a single flat layout."""
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cat_dir in sorted(raw_dir.glob("CAT_*")):
        if not cat_dir.is_dir():
            continue
        for jpg in cat_dir.glob("*.jpg"):
            cat_file = jpg.with_suffix(jpg.suffix + ".cat")
            if not cat_file.exists():
                continue
            shutil.copy2(jpg, images_dir / jpg.name)
            shutil.copy2(cat_file, annotations_dir / cat_file.name)
            n += 1
    return n


def main():
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["data_raw"]) / "crawford"
    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    annotations_dir = resolve_path(cfg["paths"]["crawford_annotations_dir"])

    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        n = len(list(images_dir.glob("*.jpg")))
        print(f"Dataset already present: {n} images in {images_dir}. Skipping.")
        return

    downloads_dir = raw_dir / "_downloads"
    for url in DATASET_URLS:
        fname = url.rsplit("/", 1)[-1]
        dest = downloads_dir / fname
        print(f"Downloading {fname} ...")
        download_file(url, dest)
        extract_zip(dest, raw_dir / "_extracted")

    n = consolidate(raw_dir / "_extracted", images_dir, annotations_dir)
    print(f"Consolidated {n} image/annotation pairs into {images_dir}")

    shutil.rmtree(downloads_dir, ignore_errors=True)
    shutil.rmtree(raw_dir / "_extracted", ignore_errors=True)


if __name__ == "__main__":
    main()
