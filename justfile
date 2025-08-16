
style:
    uv tool run --default-index https://pypi.org/simple ruff format .

format: style

check:
    uv tool run --default-index https://pypi.org/simple ruff check .

run arguments:
    time uv run --default-index https://pypi.org/simple generate_synology_folder_thumbnail.py \
        --seed 1234 \
        {{arguments}}

run-force:
    time uv run --default-index https://pypi.org/simple generate_synology_folder_thumbnail.py \
        /Volumes/photo/Famille/2025 \
        --force-image1 "/Volumes/photo/Famille/2025/25.02/2025-02-01 12.26.25.heic" \
        --force-image2 "/Volumes/photo/Famille/2025/25.04.19-20 - Week end Carcasonne/2025.04.21-14.38.16.heic" \
        --force-image3 "/Volumes/photo/Famille/2025/25.07/2025-07-05 22.11.54.heic" \
        --force-image4 "/Volumes/photo/Famille/2025/25.07/2025-07-26 23.16.49.jpg"

run-all:
    just run '/Volumes/photo/GS\ Photographie/2024/'
    just run '/Volumes/photo/Gaetan/2021'
    just run '/Volumes/photo/Gaetan/2022'
    just run '/Volumes/photo/Gaetan/2023'
    just run '/Volumes/photo/Gaetan/2024'
    just run '/Volumes/photo/Gaetan/2024'
    just run '/Volumes/photo/Gaetan/2025'
