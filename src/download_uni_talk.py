'''
Copied from UniTalk github repo for downloading the dataset, just made small change for
only downloading the val set and using curl instead of wget.
'''

import argparse
import logging
import os
import subprocess
import shutil
import time
import zipfile

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

# HuggingFace dataset base (blob form; download_and_extract / download_csv swap in /resolve/).
HF_BASE = "https://huggingface.co/datasets/plnguyen2908/UniTalk-ASD/blob/main"



def download_and_extract(blob_url: str, output_path: str):
    """Download a zip from HuggingFace (blob URL) and extract it next to output_path."""
    raw_url = blob_url.replace("/blob/", "/resolve/")
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    # clean prior
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    elif os.path.exists(output_path):
        os.remove(output_path)
    # download
    subprocess.run(["curl", "-L", "-o", output_path, raw_url], check=True)
    # unzip & delete zip
    with zipfile.ZipFile(output_path, 'r') as z:
        z.extractall(path=out_dir)
    os.remove(output_path)


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

def download_video_and_csv(video_id: str, url: str, video_dir: str, csv_dir: str,
                           split: str = "val", download_videos: bool = True):
    """Fetch a single video and its annotation CSV, skipping anything already present.

    Args:
        video_id:        YouTube id, used for the on-disk filenames.
        url:             YouTube watch URL.
        video_dir:       directory the <video_id>.mp4 file lives in.
        csv_dir:         directory the <video_id>.csv annotation file lives in.
        split:           dataset split for the HuggingFace CSV path (default 'val').
        download_videos: when False, never fetch the video (only the CSV).

    Returns:
        tuple[str, str]: (video_path, csv_path).
    """
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    csv_path = os.path.join(csv_dir, f"{video_id}.csv")

    # a) Full video
    if os.path.exists(video_path):
        logger.info(f"[skip] video already present: {video_path}")
    elif download_videos:
        logger.info(f"[download] video {video_id}")
        download_youtube_video(url, video_dir)
    else:
        logger.info(f"[skip] video download disabled, missing: {video_path}")

    # b) Annotation CSV
    if os.path.exists(csv_path):
        logger.info(f"[skip] csv already present: {csv_path}")
    else:
        logger.info(f"[download] csv {video_id}")
        download_csv(video_id, csv_dir, split=split)

    return video_path, csv_path


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
    for sub in ["csv", f"clips_videos/{split}", f"clips_audios/{split}", f"videos/{split}"]:
        os.makedirs(os.path.join(args.save_path, sub), exist_ok=True)

    start = time.time()
    with open(os.path.join("video_list", f"{split}.csv"), "r") as f:
        for line in f:
            link = line.strip()
            if not link or link == "Link":
                continue
            video_id = link.split("v=")[-1]

            # Full video + per-video annotation CSV (skips files already on disk).
            download_video_and_csv(video_id, link, video_dir, csv_dir,
                                   split=split, download_videos=args.download_videos)

            # Clip-level video / audio archives.
            for kind in ("clips_videos", "clips_audios"):
                blob_url = f"{HF_BASE}/{kind}/{split}/{video_id}.zip"
                out_zip = os.path.join(args.save_path, kind, split, f"{video_id}.zip")
                download_and_extract(blob_url, out_zip)

    logger.info(f"The whole script runs in {time.time() - start} seconds")


if __name__ == "__main__":
    main()
