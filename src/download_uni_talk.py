'''
Copied from UniTalk github repo for downloading the dataset, just made small change for
only downloading the val set and using curl instead of wget.
'''

import argparse
import logging
import os
import subprocess
import time

import pandas as pd

from helpers import setup_logging

logger = logging.getLogger("download_unitalk")

# Column names for the per-video annotation CSVs (the files on HuggingFace have no header).
CSV_COLUMNS = [
    "video_id", "frame_timestamp",
    "entity_box_x1", "entity_box_y1",
    "entity_box_x2", "entity_box_y2",
    "label", "entity_id",
    "label_id", "instance_id",
]

# HuggingFace dataset base (blob form; swapped to /resolve/ when downloading).
HF_BASE = "https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD/blob/main"


def download_youtube_video(url: str, output_dir: str) -> bool:
    """Download a full YouTube video as mp4 via yt-dlp.  Returns True on success."""
    os.makedirs(output_dir, exist_ok=True)
    result = subprocess.run([
        "yt-dlp",
        "--extractor-args", "youtube:player_client=tv_embedded",
        "-f", "bestvideo[vcodec^=avc1]+bestaudio/bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
        url,
    ])
    if result.returncode != 0:
        logger.warning(f"WARNING: failed to download {url}, skipping.")
    return result.returncode == 0


def download_csv(video_id: str, csv_dir: str, split: str = "val") -> str:
    """Download one per-video annotation CSV and rewrite it with a header row.
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{video_id}.csv")
    raw_url = f"{HF_BASE}/csv/{split}/{video_id}.csv".replace("/blob/", "/resolve/")
    subprocess.run(["curl", "-L", "-o", csv_path, raw_url], check=True)
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
    return csv_path


# ── main entry point used by the pipeline ─────────────────────────────────────

def download_video(video_id: str, url: str, video_dir: str, download_videos: bool = True):
    """Fetch a single full video, skipping it if already present.

    Args:
        video_id:        YouTube id, used for the on-disk filename.
        url:             YouTube watch URL.
        video_dir:       directory the <video_id>.mp4 file lives in.
        download_videos: when False, never fetch the video.

    Returns:
        str: video_path.
    """
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    if os.path.exists(video_path):
        logger.info(f"[skip] video already present: {video_path}")
    elif download_videos:
        logger.info(f"[download] video {video_id}")
        download_youtube_video(url, video_dir)
    else:
        logger.info(f"[skip] video download disabled, missing: {video_path}")

    return video_path


# ── standalone full-dataset download ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="where to save the dataset")
    parser.add_argument("--download_videos", action="store_true",
                        help="download full videos from YouTube links in video_list")
    parser.add_argument("--split", type=str, default="val", help="dataset split (default: val)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Also emit INFO-level messages on the console")
    args = parser.parse_args()

    setup_logging("download_unitalk", args.verbose)

    split = args.split
    video_dir = os.path.join(args.save_path, "videos", split)
    csv_dir = os.path.join(args.save_path, "csv")
    for sub in ["csv", f"videos/{split}"]:
        os.makedirs(os.path.join(args.save_path, sub), exist_ok=True)

    start = time.time()
    df_list = []
    with open(os.path.join("video_list", f"{split}.csv"), "r") as f:
        for line in f:
            link = line.strip()
            if not link or link == "Link":
                continue
            video_id = link.split("v=")[-1]

            # Full video (skips files already on disk).
            download_video(video_id, link, video_dir,
                           download_videos=args.download_videos)

            csv_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"csv/{split}/{video_id}.csv"
            )
            out_csv = os.path.join(args.save_path, "csv", f"{split}_{video_id}.csv")
            raw_csv = csv_url.replace("/blob/", "/resolve/")
            subprocess.run(["curl", "-L", "-o", out_csv, raw_csv], check=True)

            # push to list before writing into a big file
            df = pd.read_csv(out_csv, header=None, names=CSV_COLUMNS)
            df_list.append(df)

            os.remove(out_csv)

    # Merge & write {split}_orig.csv
    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        out_merged = os.path.join(args.save_path, "csv", f"{split}_orig.csv")
        merged.to_csv(out_merged, index=False)
        logger.info(f"Wrote merged CSV for {split}: {out_merged}")



if __name__ == "__main__":
    main()
