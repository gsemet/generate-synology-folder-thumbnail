
dev:
    rm -rf .venv
    python -m venv .venv
    .venv/bin/pip install -r requirements.txt

run-subfolder subfolder:
    #!/usr/bin/env bash
    set -euo pipefail
    for arg in "{{subfolder}}/"*; do
        .venv/bin/python generate-synology-forder-thumbnail.py "$arg"
    done

run argument:
    .venv/bin/python generate-synology-forder-thumbnail.py "{{argument}}"
