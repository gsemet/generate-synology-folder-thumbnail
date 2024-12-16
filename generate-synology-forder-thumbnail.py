# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=9.0",
#   "click",
#   "pillow_heif",
#   "tqdm",
#   "opencv-python",
# ]
# ///

import random
from pathlib import Path
from typing import List
from PIL import Image, ImageOps, ExifTags, ImageDraw
from pillow_heif import register_heif_opener
import click
from tqdm import tqdm


# Register HEIC support for Pillow
register_heif_opener()


def get_images_from_folder(folder: Path) -> List[Path]:
    """Recursively gets all image files from the folder and its subfolders."""
    image_files = []
    click.secho(f"Selecting 4 random pictures under {folder}:", fg="yellow")
    for file in tqdm(folder.rglob("*"), desc="Searching pictures ..."):
        if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}:
            image_files.append(file)
    return image_files


def add_margin(
    image: "Image",
    padding_top: int,
    padding_right: int,
    padding_bottom: int,
    padding_left: int,
    padding_color: int,
) -> "Image":
    width, height = image.size
    new_width = width + padding_right + padding_left
    new_height = height + padding_top + padding_bottom
    result = Image.new(image.mode, (new_width, new_height), padding_color)
    result.paste(image, (padding_left, padding_top))
    return result


def add_corners(
    image: "Image",
    radius: int,
    color=(
        255,
        255,
        255,
    ),
) -> "Image":
    """generate round corner for image"""
    width, height = image.size
    mask = Image.new("L", image.size, 0)

    draw = ImageDraw.ImageDraw(mask)
    back_color = Image.new(
        image.mode,
        image.size,
        color,
    )
    draw.rounded_rectangle(
        ((0, 0), (width, height)),
        radius=radius,
        fill=255,
    )
    return Image.composite(
        image,
        back_color,
        mask,
    )


def crop_to_aspect_ratio(
    image: "Image",
    target_width=1200,
    target_height=1200,
    padding_top=0,
    padding_right=0,
    padding_bottom=0,
    padding_left=0,
) -> "Image":
    """Crops an image to the target aspect ratio (16:9), centered."""

    click.secho(
        f"  Original size: {image.size[0]}x{image.size[1]}"
        + f" - Target size: {target_width}x{target_height}"
    )
    return add_margin(
        add_corners(
            ImageOps.fit(
                image,
                (target_width, target_height),
                method=Image.BICUBIC,
                bleed=0.0,
                centering=(0.5, 0.5),
            ),
            radius=64,
        ),
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        padding_color=(255, 255, 255),
    )


def assemble_grid(
    images: "List[Image]",
    grid_size: tuple = (2, 2),
) -> "Image":
    """Assembles 4 images into a 2x2 grid."""
    width, height = images[0].size
    grid_width = width * grid_size[1]
    grid_height = height * grid_size[0]
    grid = Image.new("RGB", (grid_width, grid_height))

    for idx, image in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid.paste(image, (col * width, row * height))

    return grid


def generate_thumbnail_grid(
    input_folder: Path,
    output_file: Path,
    target_width=1200,
    target_height=1200,
) -> None:
    """Processes images to create a 2x2 grid and save the result."""
    click.echo(f"Generating folder thumbnails for  images from: {input_folder}")
    images = get_images_from_folder(input_folder)

    # Select 4 random images from the list
    selected_images = pick_4_images(images)

    click.secho("Selected images:\n" + "\n".join(f" - {s}" for s in selected_images))

    processed_images = []
    padding = 20
    paddings = [
        # top left picture
        (0, padding, padding, 0),
        # top right picture
        (0, 0, padding, padding),
        # bottom left picture
        (padding, padding, 0, 0),
        # bottom right picture
        (padding, 0, 0, padding),
    ]

    for idx, img_path in enumerate(selected_images):
        with read_image(img_path) as img:
            click.secho(f"Opening image {idx}/4: {img_path}", fg="yellow")
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

    grid = assemble_grid(processed_images)
    grid.save(output_file, "JPEG")
    print(f"\033[92mThumbnail saved to {output_file}\033[0m")

def pick_4_images(images: List[Path]) -> List[Path]:
    """Choose 4 images from the list."""

    # Default method: select 4 random images
    selected_images = (
        random.sample(
            images,
            4,
        )
        if len(images) >= 4
        else images[:4]
    )

    return selected_images


def read_image(img_path):
    """Read the image, rotate it based on EXIF orientation if needed."""

    image = Image.open(img_path)
    try:
        orientation = 0
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = image.getexif()

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    return image


@click.command()
@click.argument(
    "input_folder",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
)
def main(input_folder: str) -> None:
    """Main CLI entry point."""
    input_path = Path(input_folder)
    output_file = input_path / "thumbnail.jpg"
    output_file.unlink(missing_ok=True)
    generate_thumbnail_grid(input_path, output_file, 1600)


if __name__ == "__main__":
    main()
