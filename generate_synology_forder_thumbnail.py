#!/usr/bin/env python

"""
generate_synology_folder_thumbnail.py
=====================================

Generate a 2×2 folder thumbnail image from pictures or videos inside a given directory.
Intended for Synology DSM folder previews, but works with any folder of images/videos.

Features:
    - Recursively scans for .jpg, .jpeg, .png, .heic, and common video files.
    - Selects four thumbnails at random (deterministic with --seed).
    - Crops each to the requested size, optionally centering on detected eyes.
    - Adds rounded corners and margins.
    - Assembles them into a 2×2 grid and saves as `thumbnail.jpg`.
    - Fast file scanning using extension filters.
    - EXIF orientation correction.
    - HEIC/HEIF image support via pillow_heif.
"""

from __future__ import annotations

import os
import random
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import click
from PIL import Image, ImageDraw, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm
import cv2
import numpy as np

# Register HEIC support for Pillow
register_heif_opener()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".insv"}

# Haar cascade for eye detection
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def _extract_video_frame(video_path: Path) -> Image.Image:
    """Extract the middle frame of a video and add film reel effect.

    Args:
        video_path: Path to a video file.

    Returns:
        PIL Image of the middle frame with top/bottom black bars.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {middle_frame_idx} from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)

    # Add film reel effect (black bars top/bottom)
    w, h = img.size
    bar_height = int(h * 0.1)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (w, bar_height)], fill="black")
    draw.rectangle([(0, h - bar_height), (w, h)], fill="black")

    return img


def _iter_image_files(folder: Path) -> Iterable[Path]:
    """Yield image paths under a folder with extension filtering.

    Args:
        folder: Root directory to scan.

    Yields:
        Paths to image files.
    """
    patterns = tuple(f"*{ext}" for ext in IMAGE_EXTS)
    generators = (folder.rglob(pat) for pat in patterns)
    for path in chain.from_iterable(generators):
        if path.suffix.lower() in IMAGE_EXTS:
            yield path


def get_images_from_folder(folder: Path) -> List[Path]:
    """Collect image files under a folder recursively.

    Args:
        folder: Root directory to scan.

    Returns:
        List of image paths.
    """
    click.secho(f"Selecting 4 random pictures under {folder}:", fg="yellow")
    return list(tqdm(_iter_image_files(folder), desc="Searching pictures ...", unit="img"))


def add_margin(
    image: Image.Image,
    padding_top: int,
    padding_right: int,
    padding_bottom: int,
    padding_left: int,
    padding_color: Tuple[int, int, int] | int,
) -> Image.Image:
    """Add padding around an image."""
    width, height = image.size
    new_width = width + padding_right + padding_left
    new_height = height + padding_top + padding_bottom
    result = Image.new(image.mode, (new_width, new_height), padding_color)
    result.paste(image, (padding_left, padding_top))
    return result


def add_corners(image: Image.Image, radius: int, color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Apply rounded corners to an image."""
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(((0, 0), (width, height)), radius=radius, fill=255)
    back = Image.new(image.mode, (width, height), color)
    return Image.composite(image, back, mask)


def crop_to_aspect_ratio(
    image: Image.Image,
    target_width: int = 1200,
    target_height: int = 1200,
    padding_top: int = 0,
    padding_right: int = 0,
    padding_bottom: int = 0,
    padding_left: int = 0,
) -> Image.Image:
    """Crop and resize image to target size, centering on eyes if detected."""
    w, h = image.size

    # Convert to grayscale for eye detection
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    eyes = EYE_CASCADE.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(eyes) >= 1:
        # Center on the first eye detected
        x, y, ew, eh = eyes[0]
        cx, cy = x + ew // 2, y + eh // 2
        crop_box = (
            max(0, cx - target_width // 2),
            max(0, cy - target_height // 2),
            min(w, cx + target_width // 2),
            min(h, cy + target_height // 2),
        )
        image = image.crop(crop_box)

    fitted = ImageOps.fit(image, (target_width, target_height), method=Image.BICUBIC, bleed=0.0, centering=(0.5, 0.5))
    rounded = add_corners(fitted, radius=64)
    return add_margin(
        rounded,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        padding_color=(255, 255, 255),
    )


def assemble_grid(images: Sequence[Image.Image], grid_size: Tuple[int, int] = (2, 2)) -> Image.Image:
    """Assemble images into a fixed grid."""
    rows, cols = grid_size
    width, height = images[0].size
    grid = Image.new("RGB", (width * cols, height * rows))
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(image, (col * width, row * height))
    return grid


def pick_4_images(image_paths, target_width, target_height):
    """Pick four thumbnails randomly, one per quarter, keeping order."""
    if not image_paths:
        return []

    quarter_size = max(1, len(image_paths) // 4)
    selected = []
    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else len(image_paths)
        quarter_files = image_paths[start_idx:end_idx]
        if quarter_files:
            chosen_file = random.choice(quarter_files)
            ext = chosen_file.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                img = _extract_video_frame(chosen_file)
            else:
                img = Image.open(chosen_file)
            selected.append((img, chosen_file))
    return selected


def generate_thumbnail_grid(input_folder: Path, output_file: Path, target_width: int = 1200, target_height: int = 1200) -> None:
    """Create a 2x2 folder thumbnail grid from four images/videos."""
    click.secho(f"Generating folder thumbnails from {input_folder}", fg="yellow")
    images = get_images_from_folder(input_folder)
    if not images:
        click.secho(f"No images found in {input_folder}, skipping.", fg="red")
        return

    selected_images = pick_4_images(images, target_width, target_height)
    click.secho("Selected images:\n" + "\n".join(f" - {s[1]}" for s in selected_images))

    processed_images: List[Image.Image] = []
    padding = 20
    paddings = [(0, padding, padding, 0), (0, 0, padding, padding), (padding, padding, 0, 0), (padding, 0, 0, padding)]

    for idx, (img, img_path) in enumerate(selected_images):
        if img_path is None:
            click.secho(f"Slot {idx + 1}/4: empty (white box)", fg="yellow")
            processed_images.append(Image.new("RGB", (target_width, target_height), color=(255, 255, 255)))
            continue

        click.secho(f"Processing image {idx + 1}/4: {img_path}", fg="yellow")
        try:
            cropped_img = crop_to_aspect_ratio(
                img,
                target_width=target_width,
                target_height=target_height,
                padding_top=paddings[idx][0],
                padding_right=paddings[idx][1],
                padding_bottom=paddings[idx][2],
                padding_left=paddings[idx][3],
            )
            processed_images.append(cropped_img)
        finally:
            img.close()

    grid = assemble_grid(processed_images)
    grid.save(output_file, "JPEG", quality=90, optimize=True)
    click.secho(f"Thumbnail saved to {output_file}", fg="green")


def _find_image_folders(root: Path) -> Iterable[Path]:
    """Yield folders that contain subfolders but no top-level images (except thumbnail.*)."""
    scanned = 0
    click.secho(f"Scanning folders under {root} ...", fg="cyan")
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        if folder.name.startswith("."):
            continue

        scanned += 1
        if scanned % 100 == 0:
            click.secho(f"  Scanned {scanned} folders so far ...", fg="cyan")

        # Must have at least one subfolder
        if not dirnames:
            continue

        top_level_images = [
            f for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS and not Path(f).stem.lower().startswith("thumbnail")
        ]
        has_thumbnails = any(Path(f).stem.lower().startswith("thumbnail") for f in filenames)

        if not top_level_images or has_thumbnails:
            yield folder


@click.command()
@click.argument(
    "input_folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--width", type=int, default=1600, show_default=True, help="Per-tile width in pixels.")
@click.option("--height", type=int, default=1600, show_default=True, help="Per-tile height in pixels.")
def main(input_folder: Path, width: int, height: int) -> None:
    """CLI entry point."""
    for folder in _find_image_folders(input_folder):
        output_file = folder / "thumbnail.jpg"
        output_file.unlink(missing_ok=True)
        generate_thumbnail_grid(folder, output_file, target_width=width, target_height=height)


if __name__ == "__main__":
    main()
