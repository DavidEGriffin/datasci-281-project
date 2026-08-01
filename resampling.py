import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np


TOTAL = 70_000

CLASSES = (
    "city street",
    "gas stations",
    "highway",
    "parking lot",
    "residential",
    "tunnel",
)

IMAGE_DIR = Path("./images/train")
LABEL_DIR = Path("./labels/train")

OUT_IMAGE_DIR = Path("./resampled/images")
OUT_LABEL_DIR = Path("./resampled/labels")

CROP_SIZE = (600, 1000)  # (height, width)
LOG_EVERY = 100


def get_scene(label_file: Path) -> str:
    with open(label_file, "r", encoding="utf-8") as f:
        return json.load(f)["attributes"]["scene"]


def random_crop(
    image: np.ndarray,
    crop_size: tuple[int, int],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    H, W = image.shape[:2]
    crop_h, crop_w = crop_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} exceeds image size {(H, W)}."
        )

    top = rng.integers(0, H - crop_h + 1)
    left = rng.integers(0, W - crop_w + 1)

    return image[top : top + crop_h, left : left + crop_w]


def main():
    rng = np.random.default_rng(0)

    OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build mapping from class -> [(image_path, label_path), ...]
    # ------------------------------------------------------------------

    by_class: dict[str, list[tuple[Path, Path]]] = defaultdict(list)

    print("Scanning dataset...")

    for label_path in LABEL_DIR.glob("*.json"):
        scene = get_scene(label_path)

        if scene not in CLASSES:
            continue

        image_path = IMAGE_DIR / (label_path.stem + ".jpg")

        if image_path.exists():
            by_class[scene].append((image_path, label_path))

    print()

    for cls in CLASSES:
        print(f"{cls:15s}: {len(by_class[cls])} source images")

    print()

    n_per_class = TOTAL // len(CLASSES)

    total_written = 0
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Sample each class independently
    # ------------------------------------------------------------------

    for scene in CLASSES:
        candidates = by_class[scene]

        if len(candidates) == 0:
            raise RuntimeError(f"No images found for class '{scene}'")

        sampled_indices = rng.integers(
            0,
            len(candidates),
            size=n_per_class,
        )

        print(f"\nProcessing class '{scene}'...")

        for sample_num, idx in enumerate(sampled_indices):
            image_path, label_path = candidates[idx]

            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Failed to read {image_path}")

            crop = random_crop(image, CROP_SIZE, rng)

            out_stem = f"{image_path.stem}_{sample_num:05d}"

            cv2.imwrite(
                str(OUT_IMAGE_DIR / f"{out_stem}{image_path.suffix}"),
                crop,
            )

            shutil.copy2(
                label_path,
                OUT_LABEL_DIR / f"{out_stem}.json",
            )

            # Explicitly release memory before the next iteration.
            del image
            del crop

            total_written += 1

            if total_written % LOG_EVERY == 0:
                elapsed = time.perf_counter() - start_time
                rate = total_written / elapsed

                print(
                    f"[{total_written:6d}/{TOTAL}] "
                    f"{rate:6.1f} images/sec   "
                    f"elapsed {elapsed:7.1f}s"
                )

    elapsed = time.perf_counter() - start_time

    print("\nFinished.")
    print(f"Generated : {total_written:,} images")
    print(f"Elapsed   : {elapsed:.1f} s")
    print(f"Average   : {total_written / elapsed:.1f} images/sec")


if __name__ == "__main__":
    main()