#!/usr/bin/env python3
from pathlib import Path

from pytube import YouTube

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "video_list" / "val.csv"
OUTPUT_DIR = ROOT / "data" / "videos" / "val"


def get_links() -> list[str]:
    links = []
    for line in CSV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line == "Link":
            continue
        links.append(line)
    return links


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing file: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in get_links():
        video_id = url.split("v=")[-1]
        output_path = OUTPUT_DIR / f"{video_id}.mp4"

        if output_path.exists():
            print(f"Skipping {url} — {output_path.name} already exists")
            continue

        print(f"Downloading {url}")
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=str(OUTPUT_DIR))

    print(f"Done. Files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
