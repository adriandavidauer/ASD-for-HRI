#!/usr/bin/env python3
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import ExtractorError, DownloadError

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


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if video_id:
            return video_id

    if parsed.netloc == "youtu.be":
        video_id = parsed.path.lstrip("/")
        if video_id:
            return video_id

    raise ValueError(f"Could not extract YouTube video ID from URL: {url}")


def download_video(url: str) -> None:
    video_id = extract_video_id(url)
    output_path = OUTPUT_DIR / f"{video_id}.mp4"

    if output_path.exists():
        print(f"Skipping {url} — {output_path.name} already exists")
        return

    print(f"Downloading {url}")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(OUTPUT_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except (ExtractorError, DownloadError) as exc:
        message = str(exc)
        if "private video" in message.lower():
            print(f"Skipping {url} — private video")
            return
        raise


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing file: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in get_links():
        download_video(url)

    print(f"Done. Files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
