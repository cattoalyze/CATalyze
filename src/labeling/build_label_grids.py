"""Build numbered image grids for batch visual mood labeling.

Used for the seed-labeling pass documented in the README as provenance='ai'
(no human rater was available in this session — see README limitations).
Each grid image is saved with an accompanying manifest mapping cell index to
source image path, so labels assigned while viewing a grid can be recorded
programmatically afterward.
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_config, resolve_path  # noqa: E402


def build_grid(image_paths: list[Path], out_path: Path, cols: int, rows: int, cell_size: int = 190):
    n = len(image_paths)
    assert n <= cols * rows
    pad = 4
    label_h = 22
    cell_total_h = cell_size + label_h
    canvas = np.full((rows * cell_total_h + pad, cols * cell_size + pad, 3), 255, dtype=np.uint8)

    for i, img_path in enumerate(image_paths):
        r, c = divmod(i, cols)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (cell_size, cell_size))
        y0 = r * cell_total_h
        x0 = c * cell_size
        canvas[y0 : y0 + cell_size, x0 : x0 + cell_size] = img
        cv2.putText(
            canvas, str(i), (x0 + 4, y0 + cell_size + 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main(n_images: int, cols: int, rows: int, out_dir: str, seed: int):
    cfg = load_config()
    images_dir = resolve_path(cfg["paths"]["crawford_images_dir"])
    all_images = sorted(images_dir.glob("*.jpg"))

    rng = random.Random(seed)
    sampled = rng.sample(all_images, min(n_images, len(all_images)))

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    per_grid = cols * rows
    manifest = {}
    grid_idx = 0
    for start in range(0, len(sampled), per_grid):
        batch = sampled[start : start + per_grid]
        grid_path = out_dir_path / f"grid_{grid_idx:03d}.png"
        build_grid(batch, grid_path, cols, rows)
        manifest[grid_path.name] = [str(p) for p in batch]
        grid_idx += 1

    with open(out_dir_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built {grid_idx} grids ({len(sampled)} images) in {out_dir_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=600)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="data/processed/label_grids")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.n_images, args.cols, args.rows, args.out_dir, args.seed)
