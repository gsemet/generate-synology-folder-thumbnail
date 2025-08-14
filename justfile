
run-subfolder subfolder:
    #!/usr/bin/env bash
    set -euo pipefail
    for arg in "{{subfolder}}/"*; do
        .venv/bin/python generate_synology_folder_thumbnail.py "$arg"
    done

style:
    uv tool run --default-index https://pypi.org/simple ruff format .

format: style

check:
    uv tool run --default-index https://pypi.org/simple ruff check .

run-uv arguments:
    uv run --default-index https://pypi.org/simple generate_synology_forder_thumbnail.py --seed 1234 {{arguments}}

run arguments:
    .venv/bin/python generate_synology_forder_thumbnail.py "{{arguments}}"

run-all:
    just run-uv '/Volumes/photo/GS\ Photographie/2024/'
    just run-uv '/Volumes/photo/Gaetan/2024'
    just run-uv '/Volumes/photo/Gaetan/2025'
