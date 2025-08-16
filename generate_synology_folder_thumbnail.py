#!/usr/bin/env python

"""
generate_synology_folder_thumbnail.py
=====================================

Generate a 2×2 folder thumbnail image from pictures or videos inside a given directory.
Intended for Synology DSM folder previews, but works with any folder of images/videos.

Features:
    - Recursively scans for .jpg, .jpeg, .png, .heic, and common video files.
    - Selects four thumbnails at random, picking in a small selection and trying
      to rank them by relevance using locally run open source AI models.
    - Crops each to the requested size, optionally centering on detected eyes.
    - Adds rounded corners and margins.
    - Assembles them into a 2×2 grid and saves as `thumbnail.jpg`, so that
      Synology DSM can use it as a folder preview.
    - Fast file scanning using extension filters.
    - EXIF orientation correction.
    - HEIC/HEIF image support.
    - Extract thumbnail images from videos if selected.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#    "click",
#    "mediapipe",
#    "open_clip_torch",
#    "opencv-python",
#    "pillow_heif",
#    "pillow>=9.0",
#    "rawpy",
#    "torch",
#    "tqdm",
# ]
# ///

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import random

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import click
import cv2
import mediapipe as mp
import numpy as np
import open_clip
import torch

from PIL import Image, ImageDraw, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm


# Register HEIC support for Pillow
register_heif_opener()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}
IMAGE_EXTS_TUPLES = tuple(ext.lower() for ext in IMAGE_EXTS)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".insv"}

DEFAULT_CANDIDATES_PER_TILE = 5
DEFAULT_THUMBNAIL_WIDTH = 640
DEFAULT_THUMBNAIL_HEIGHT = 640
DEFAULT_THUMBNAIL_FILENAME = "thumbnail.jpg"
DEFAULT_FACE_MARGIN_RATIO = 0.2
FACE_VERTICAL_SHIFT_RATIO = -0.1
DEFAULT_CORNER_RADIUS_PERCENT = 0.04

# Haar cascade for eye detection
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Initialize Mediapipe face detector
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize OpenCLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
)
clip_model.to(device)
clip_model.eval()

CLIP_PROMPTS = [
    # People – group-focused
    "a group of people smiling and looking at the camera outdoors",
    "friends or family gathered together in a happy moment",
    "a wedding party or celebration with many people",
    "a team of people posing for a picture",
    # People – reduced emphasis on single faces
    "a person in a natural setting with an interesting background",
    "a candid shot of a person interacting with others",
    # Landscape / scenery
    "a wide landscape with mountains and rivers under dramatic light",
    "a colorful sunset over a city skyline",
    "a forest with sunlight streaming through the trees",
    "an open field with a vivid sky",
    # Composition / aesthetics
    "a well-composed photograph with balanced framing",
    "an image with sharp details and pleasing depth",
    "a scene with symmetry and harmonious colors",
    "a photograph with strong leading lines and perspective",
    # Action / emotion
    "people sharing joy in a lively environment",
    "a dynamic action scene with motion",
    "a meaningful moment between multiple people",
]

"""
CLIP_PROMPTS: List of textual prompts used to rank images by 'interest'.

Each prompt describes a type of scene, subject, or aesthetic quality that makes
an image visually engaging. When using a CLIP-based model, images are scored
against these prompts, and higher scores indicate images that are more likely
to be interesting or appealing.

Categories covered:
    - People: emphasizes faces, groups, emotions, and interactions.
    - Landscape / scenery: highlights wide, colorful, or dramatic natural or urban scenes.
    - Composition / aesthetics: favors well-composed, balanced, and visually striking images.
    - Action / emotion: rewards dynamic moments or expressive behavior.

Example usage:
    scores = [clip_model.score(img, CLIP_PROMPTS) for img in images]
    best_image = images[np.argmax(scores)]
"""
TEXT_TOKENS = open_clip.tokenize(CLIP_PROMPTS).to(device)
with torch.no_grad():
    TEXT_FEATURES = clip_model.encode_text(TEXT_TOKENS)
    TEXT_FEATURES /= TEXT_FEATURES.norm(dim=-1, keepdim=True)


@lru_cache(maxsize=4096)
def _cached_walk_dir(path: str) -> Tuple[List[str], List[str]]:
    dirs: List[str] = []
    files: List[str] = []
    try:
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False):
                files.append(entry.name)
    except PermissionError:
        return [], []
    return dirs, files


def _iter_walk(root: str, pbar=None):
    dirs, files = _cached_walk_dir(root)
    if pbar is not None:
        pbar.update(1)  # update for each visited directory
    yield root, dirs, files
    for d in dirs:
        subpath = os.path.join(root, d)
        yield from _iter_walk(subpath, pbar)


def cached_walk(root: str | Path, show_progress: bool = False):
    pbar = tqdm(desc="Walking folders", unit=" dir") if show_progress else None
    try:
        yield from _iter_walk(str(root), pbar)
    finally:
        if pbar is not None:
            pbar.close()


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


def _iter_image_files(folder: Path, pbar=None) -> Iterable[Path]:
    """Yield image paths under a folder with extension filtering."""
    stack = [str(folder)]
    dir_count = 0
    img_count = 0
    dir_batch = 12  # update pbar every N dirs
    img_batch = 50  # update postfix every N images

    while stack:
        path = stack.pop()
        dir_count += 1
        if pbar is not None and dir_count % dir_batch == 0:
            pbar.update(dir_batch)

        try:
            with os.scandir(path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        name = entry.name
                        # Faster than splitext -> lowercase -> in set
                        if name.lower().endswith(IMAGE_EXTS_TUPLES):
                            img_count += 1
                            if pbar is not None and img_count % img_batch == 0:
                                pbar.set_postfix(images=img_count)
                            yield Path(entry.path)
        except PermissionError:
            continue

    if pbar is not None:
        pbar.update(dir_count % dir_batch)  # final leftover
        pbar.set_postfix(images=img_count)  # final count


def get_images_from_folder(folder: Path) -> List[Path]:
    """Collect image files under a folder recursively."""
    click.secho(f"Selecting 4 random pictures under {folder}...", fg="yellow")
    with tqdm(
        desc="Scanning folders",
        unit=" dir",
    ) as pbar:
        return list(_iter_image_files(folder, pbar))


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


def add_corners(
    image: Image.Image,
    radius: int,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Apply rounded corners to an image."""
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        ((0, 0), (width, height)),
        radius=radius,
        fill=255,
    )
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
    face_margin_ratio: float = DEFAULT_FACE_MARGIN_RATIO,
) -> Image.Image:
    """
    Crop and resize the image to the target aspect ratio, centering on detected faces if any.
    If multiple faces are detected, it computes a bounding box around all significant faces.
    Respects EXIF rotation metadata.

    Steps:
    - Apply EXIF transpose to respect orientation.
    - Detect faces and get their bounding boxes.
    - If one or more faces are found, compute a group bounding box and center the crop there.
    - Maintain the target aspect ratio during cropping (no squeezing).
    - Add rounded corners and optional padding.

    Parameters:
        image (PIL.Image.Image): The source image.
        target_width (int): Final width in pixels.
        target_height (int): Final height in pixels.
        padding_top, padding_right, padding_bottom, padding_left (int): Padding in pixels.

    Returns:
        PIL.Image.Image: Cropped, resized image with optional padding.
    """
    # Apply EXIF-based rotation/mirroring
    image = ImageOps.exif_transpose(image)
    w, h = image.size

    img_np = np.array(image.convert("RGB"))
    results = face_detector.process(img_np)  # assumes global face_detector

    # Default center is image center
    cx, cy = w // 2, h // 2

    if results.detections:
        # Collect all face centers and bounding boxes
        face_boxes = []
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            face_boxes.append((x1, y1, x1 + bw, y1 + bh))

        # Compute group bounding box
        min_x = min(b[0] for b in face_boxes)
        min_y = min(b[1] for b in face_boxes)
        max_x = max(b[2] for b in face_boxes)
        max_y = max(b[3] for b in face_boxes)

        group_w = max_x - min_x
        group_h = max_y - min_y

        # Expand with margin
        margin_x = int(group_w * face_margin_ratio)
        margin_y = int(group_h * face_margin_ratio)
        min_x = max(0, min_x - margin_x)
        min_y = max(0, min_y - margin_y)
        max_x = min(w, max_x + margin_x)
        max_y = min(h, max_y + margin_y)

        # Updated group center
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2

        # Shift upward (reduce cy)
        shift_up = int(group_h * DEFAULT_FACE_MARGIN_RATIO)
        cy = max(0, cy - shift_up)

        group_w = max_x - min_x
        group_h = max_y - min_y
    else:
        group_w = w
        group_h = h

    # Maintain target aspect ratio
    target_aspect = target_width / target_height
    crop_width = group_w
    crop_height = group_h

    if crop_width / crop_height < target_aspect:
        crop_width = int(crop_height * target_aspect)
    else:
        crop_height = int(crop_width / target_aspect)

    # Center crop around cx, cy
    left = max(0, cx - crop_width // 2)
    top = max(0, cy - crop_height // 2)
    right = min(w, left + crop_width)
    bottom = min(h, top + crop_height)

    # Adjust if crop goes out of bounds
    if right - left < crop_width:
        left = max(0, right - crop_width)
    if bottom - top < crop_height:
        top = max(0, bottom - crop_height)

    cropped = image.crop((left, top, right, bottom))

    # Resize to final output
    fitted = ImageOps.fit(cropped, (target_width, target_height), method=Image.BICUBIC)
    radius = int(fitted.width * DEFAULT_CORNER_RADIUS_PERCENT)
    rounded = add_corners(fitted, radius=radius)
    return add_margin(
        rounded,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        padding_color=(255, 255, 255),
    )


def assemble_grid(
    images: Sequence[Image.Image],
    grid_size: Tuple[int, int] = (2, 2),
) -> Image.Image:
    """Assemble images into a fixed grid."""
    rows, cols = grid_size
    width, height = images[0].size
    grid = Image.new("RGB", (width * cols, height * rows))
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(image, (col * width, row * height))
    return grid


def score_image(path: Path) -> float:
    """Return an 'interestingness' score using Mediapipe + OpenCLIP.

    The score is calculated based on the following components:
    - **Face Detection (Mediapipe)**:
        - Number of detected faces (each adds a base score).
        - Proximity of faces to the image center (closer faces add more score).
    - **Semantic Similarity (OpenCLIP)**:
        - Measures how closely the image matches a set of predefined textual prompts.
    - **Sharpness**:
        - Estimated using the variance of the Laplacian of the grayscale image.
    - **Colorfulness**:
        - Quantified using the difference between RGB channels.

    Parameters:
        path (Path): Path to the image file.

    Returns:
        float: A composite score representing the image's overall 'interestingness'.
            Returns 0.0 if the image cannot be loaded or processed.

    """
    try:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
    except Exception:
        return 0.0

    score = 0.0
    h, w, _ = img_np.shape

    # Face detection (Mediapipe)
    results = face_detector.process(img_np)
    if results.detections:
        # More faces = higher score
        score += len(results.detections) * 10
        # Face near center bonus
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            cx = (bbox.xmin + bbox.width / 2) * w
            cy = (bbox.ymin + bbox.height / 2) * h
            dist_to_center = np.linalg.norm(
                np.array([cx, cy]) - np.array([w / 2, h / 2])
            )
            score += max(0, 10 - (dist_to_center / max(w, h) * 20))

    # OpenCLIP semantic scoring
    try:
        img_input = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(img_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ TEXT_FEATURES.T).squeeze(0)
            score += similarities.max().item() * 20  # weight semantic similarity
    except Exception:
        pass

    # Improved sharpness scoring
    img_gray = np.array(img.convert("L"))
    lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_score = np.mean(sobel_mag)

    if lap_var < 100 and sobel_score < 10:
        score -= 10  # Penalize blurry images
    else:
        score += min(10, (lap_var + sobel_score) / 200)

    # Colorfulness bonus
    (R, G, B) = np.array(img).astype("float").transpose(2, 0, 1)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    colorfulness = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2) + 0.3 * (
        rg.std() + yb.std()
    )
    score += colorfulness / 20

    img.close()
    return score


def pick_4_images(
    image_paths,
    rng: random.Random,
    candidates_per_tile: int,
):
    """Pick one thumbnail per tile, preferring images with people and sharpness."""
    if not image_paths:
        return []

    quarter_size = max(1, len(image_paths) // 4)
    selected = []

    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else len(image_paths)
        quarter_files = image_paths[start_idx:end_idx]
        if not quarter_files:
            continue

        # pick candidates_per_tile random candidates per quarter
        candidates = rng.sample(
            quarter_files,
            min(candidates_per_tile, len(quarter_files)),
        )
        click.secho(
            f"  Tile {i}/4: Candidates...\n"
            + "\n".join(f"    - {str(c)}" for c in candidates),
            fg="yellow",
        )
        click.secho(
            f"  Tile {i}/4: Ranking...",
            fg="yellow",
        )

        # Parallel scoring
        with ProcessPoolExecutor() as executor:
            scores = list(executor.map(score_image, candidates))

        scored_candidates = list(zip(scores, candidates))

        click.secho(
            f"  Tile {i}/4: Scored candidates:\n"
            + "\n".join(
                f"    - {s[0]:>6.2f}: {s[1]}"
                for s in sorted(
                    scored_candidates,
                    key=lambda x: x[0],
                    reverse=True,
                )
            ),
            fg="yellow",
        )
        # select the one with the highest score
        best_path = max(scored_candidates, key=lambda x: x[0])[1]
        click.secho(
            f"  Tile {i}/4: Selected {best_path}",
            fg="green",
        )

        ext = best_path.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            img = _extract_video_frame(best_path)
        else:
            img = Image.open(best_path)

        selected.append((img, best_path))

    return selected


def generate_thumbnail_grid(
    input_folder: Path,
    output_file: Path,
    rng: random.Random,
    target_width: int = 1200,
    target_height: int = 1200,
    candidates_per_tile: int = DEFAULT_CANDIDATES_PER_TILE,
    force_image1: Path | None = None,
    force_image2: Path | None = None,
    force_image3: Path | None = None,
    force_image4: Path | None = None,
) -> None:
    """Create a 2x2 folder thumbnail grid from four images/videos."""
    click.secho(f"Generating folder thumbnails from {input_folder}", fg="yellow")

    if force_image1 and force_image2 and force_image3 and force_image4:
        selected_images = [
            (Image.open(force_image1), force_image1),
            (Image.open(force_image2), force_image2),
            (Image.open(force_image3), force_image3),
            (Image.open(force_image4), force_image4),
        ]
        click.secho(
            "Forced images:\n" + "\n".join(f" - {s[1]}" for s in selected_images),
            fg="yellow",
        )
    else:
        images = get_images_from_folder(input_folder)
        if not images:
            click.secho(f"No images found in {input_folder}, skipping.", fg="red")
            return
        selected_images = pick_4_images(
            images,
            rng=rng,
            candidates_per_tile=candidates_per_tile,
        )
    click.secho("Selected images:\n" + "\n".join(f" - {s[1]}" for s in selected_images))

    processed_images: List[Image.Image] = []
    padding_percent = 0.0125  # 1.25%
    padding = int(target_width * padding_percent)

    paddings = [
        (0, padding, padding, 0),
        (0, 0, padding, padding),
        (padding, padding, 0, 0),
        (padding, 0, 0, padding),
    ]

    for idx, (img, img_path) in enumerate(selected_images):
        if img_path is None:
            click.secho(f"Slot {idx + 1}/4: empty (white box)", fg="yellow")
            processed_images.append(
                Image.new("RGB", (target_width, target_height), color=(255, 255, 255))
            )
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
    for dirpath, dirnames, filenames in cached_walk(
        root,
        show_progress=False,
    ):
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
            f
            for f in filenames
            if Path(f).suffix.lower() in IMAGE_EXTS
            and not Path(f).stem.lower().startswith("thumbnail")
        ]
        has_thumbnails = any(
            Path(f).stem.lower().startswith("thumbnail") for f in filenames
        )

        if not top_level_images or has_thumbnails:
            yield folder


@click.command()
@click.argument(
    "input_folder",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--width",
    type=int,
    default=DEFAULT_THUMBNAIL_WIDTH,
    show_default=True,
    help="Per-tile width in pixels.",
)
@click.option(
    "--height",
    type=int,
    default=DEFAULT_THUMBNAIL_HEIGHT,
    show_default=True,
    help="Per-tile height in pixels.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible output.",
)
@click.option(
    "--candidates-per-tile",
    type=int,
    default=DEFAULT_CANDIDATES_PER_TILE,
    show_default=True,
    help="Number of candidate images to consider per tile when ranking.",
)
@click.option(
    "--force-image1",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the first image to force.",
)
@click.option(
    "--force-image2",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the second image to force.",
)
@click.option(
    "--force-image3",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the third image to force.",
)
@click.option(
    "--force-image4",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
    help="Path to the fourth image to force.",
)
def main(
    input_folder: Path,
    width: int,
    height: int,
    seed: int,
    candidates_per_tile: int,
    force_image1: Path | None = None,
    force_image2: Path | None = None,
    force_image3: Path | None = None,
    force_image4: Path | None = None,
) -> None:
    """CLI entry point."""
    rng = random.Random(seed)
    click.secho("Starting thumbnail generation...", fg="magenta")
    click.secho(f"Input folder: {input_folder}", fg="cyan")
    click.secho(f"Target size: {width}x{height}", fg="cyan")
    click.secho(f"Random seed: {seed}", fg="cyan")
    click.secho(f"Candidates per tile: {candidates_per_tile}", fg="cyan")

    if force_image1 and force_image2 and force_image3 and force_image4:
        output_file = input_folder / DEFAULT_THUMBNAIL_FILENAME
        output_file.unlink(missing_ok=True)
        generate_thumbnail_grid(
            input_folder,
            output_file,
            target_width=width,
            target_height=height,
            rng=rng,
            candidates_per_tile=candidates_per_tile,
            force_image1=force_image1,
            force_image2=force_image2,
            force_image3=force_image3,
            force_image4=force_image4,
        )
        return

    for folder in _find_image_folders(input_folder):
        output_file = folder / DEFAULT_THUMBNAIL_FILENAME
        output_file.unlink(missing_ok=True)
        generate_thumbnail_grid(
            folder,
            output_file,
            target_width=width,
            target_height=height,
            rng=rng,
            candidates_per_tile=candidates_per_tile,
        )


if __name__ == "__main__":
    main()
