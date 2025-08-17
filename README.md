# generate_synology_folder_thumbnail.py

Generate a 2×2 folder thumbnail image from pictures or videos inside a given directory.
Designed for Synology DSM folder previews, but works with any folder containing images or videos.

It is mean to provide to Synology Photos (or File Station or  Video Station / Media Server)
the thumbnail of folder without direct photo, for example in a hierarchical folder structure
is used.

It is also a playground of mine to evaluate local AI model usage, using CLIP and OpenCV.

## Rationale

I use a deeply nested hierarchical structure that worked for years, and compatible with
both my "pseudo-professional" photography life, hobbies and family life.

The structure is like this:

```text
/volume3/photo/
├── Family
│ ├── 2023
│ │ ├── 23.05 - Sophie’s wedding
│ │ └── 23.12 - Christmas at grandparents
│ ├── 2024
│ │ ├── 24.02 - Ski holidays
│ │ └── 24.08 - Lucas’s birthday
│ └── 2025
│ ├── 25.04 - Easter in Bordeaux
│ └── 25.08 - Holidays in Creusot
├── Professional
│ ├── 2023
│ │ ├── 23.03 - Data conference Paris
│ │ └── 23.09 - X200 product launch
│ ├── 2024
│ │ ├── 24.01 - AutoTech trade show
│ │ └── 24.11 - Internal training
│ └── 2025
│ ├── 25.05 - Mission in Berlin
│ └── 25.07 - Security workshop
└── Personal
  ├── 2023
  │ ├── 23.04 - Hiking in Brittany
  │ └── 23.10 - Jazz festival
  ├── 2024
  │ ├── 24.03 - Home project
  │ └── 24.09 - Trip to Japan
  └── 2025
    ├── 25.02 - Photography workshop
    └── 25.06 - Road trip Portugal
```

Synology Photos works with this, but does not create a thumbnail for folders that
does not contain at least one picture.

Using this program from the top level (ex: `/volume3/photo/`), all intermediate levels
(`/volume3/photo/Family`, `/volume3/photo/Family/2023`,...) will have a file
name `thumbnail.jpeg`.

It is also a way to learn more about AI, face recognition projects and so one.

## Features

- Recursively scans for supported image and video formats
  (`.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.tiff`, `.bmp`, `.mov`, `.mp4`, `.m4v`, `.avi`, `.mkv`, `.insv`)
- Picks four images or video frames per folder
  - Uses random sampling with ranking by:
    - Faces detected with Mediapipe
    - Semantic similarity to CLIP prompts
    - Sharpness and colorfulness
- Crops to a target size, optionally centering on detected faces
- Adds rounded corners and margins
- Saves the result as `thumbnail.jpg`
- HEIC/HEIF image support
- EXIF orientation correction
- Fast directory walking with caching

## How it works

- Scans each folder for supported media
- Selects candidates and scores them using:
  - Face detection and positioning
  - CLIP-based similarity to predefined prompts
  - Sharpness (Laplacian variance)
  - Colorfulness
- Crops and formats the four best images
- Combines them into a 2×2 grid saved as thumbnail.jpg

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for easy execution and dependency management
- Dependencies:
  - click
  - mediapipe
  - open_clip_torch
  - opencv-python
  - pillow_heif
  - pillow>=9.0
  - rawpy
  - torch
  - tqdm

Note: This will download some opensource AI model and execute locally. It works fine on my
MacBookPro with synology Photos folder mounted as a SMB share, but i did not tested on any other
configuration.

Also, the models might be huge, several hundred MB ! So the script is small, but its dependencies
are huge ! But thanks to `uv`, this is completely abstracted.

## Installation

Install [uv](https://docs.astral.sh/uv/), then run the script without manually installing dependencies:

```bash
uv run generate_synology_folder_thumbnail.py /path/to/synology/photo/root/
```

`uv` will create an isolated environment and install required dependencies automatically.

## Options

- `--width`: Per-tile width in pixels (default: 1600)
- `--height`: Per-tile height in pixels (default: 1600)
- `--candidates-per-tile`: Number of candidate images to consider per tile before ranking (default: 5)

Debugging and Reproducibility

- `--seed`: Set a random seed for reproducible output
- `--force-image1` ... `--force-image4`: Force specific images for each slot in the grid

## Output layout

The script produces a 2×2 JPEG thumbnail named `thumbnail.jpg`:

```text
+-----------+-----------+
|   Tile 1  |   Tile 2  |
+-----------+-----------+
|   Tile 3  |   Tile 4  |
+-----------+-----------+
```
