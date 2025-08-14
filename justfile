
dev:
    rm -rf .venv
    python -m venv .venv
    .venv/bin/pip install --index-url https://pypi.org/simple -r requirements.txt

run-subfolder subfolder:
    #!/usr/bin/env bash
    set -euo pipefail
    for arg in "{{subfolder}}/"*; do
        .venv/bin/python generate_synology_folder_thumbnail.py "$arg"
    done

style:
    uv tool run ruff format .

check:
    uv tool run ruff check .

run-uv arguments:
    uv run --default-index https://pypi.org/simple generate_synology_forder_thumbnail.py {{arguments}}

run arguments:
    .venv/bin/python generate_synology_forder_thumbnail.py "{{arguments}}"
