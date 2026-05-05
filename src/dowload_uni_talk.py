'''
Copied from UniTalk github repo for downloading the dataset, just made small change for only downloading val set and using curl instead of wget
'''

import argparse
import os
import subprocess
import shutil
import zipfile
import pandas as pd
import time

# 1) Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, required=True, help="where to save the dataset")
parser.add_argument("--download_videos", action="store_true", help="download full videos from YouTube links in video_list")
args = parser.parse_args()

# 2) Prepare folders
subdirs = [
    "csv",
    "clips_videos/train", "clips_videos/val",
    "clips_audios/train", "clips_audios/val",
    "videos/train", "videos/val"
]
for sub in subdirs:
    os.makedirs(os.path.join(args.save_path, sub), exist_ok=True)

# 3) Utility to download & unzip
def download_and_extract(blob_url: str, output_path: str):
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

# 4) Download full video from YouTube link using yt-dlp
def download_youtube_video(url: str, output_dir: str):
    result = subprocess.run(
        [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=tv_embedded",
            "-o", os.path.join(output_dir, "%(id)s.%(ext)s"),
            url,
        ]
    )
    if result.returncode != 0:
        print(f"WARNING: failed to download {url}, skipping.")

# 5) Column names (no header in per-video CSVs)
columns = [
    "video_id", "frame_timestamp",
    "entity_box_x1", "entity_box_y1",
    "entity_box_x2", "entity_box_y2",
    "label", "entity_id",
    "label_id", "instance_id"
]

start = time.time()
# 5) Process each split separately
for split in ["val"]:
    df_list = []

    with open(os.path.join("video_list", f"{split}.csv"), "r") as f:
        for line in f:
            link = line.strip()
            if not link or link == "Link":
                continue
            video_name = link.split("v=")[-1]

            # a) Full video download from YouTube
            if args.download_videos:
                videos_dir = os.path.join(args.save_path, "videos", split)
                download_youtube_video(link, videos_dir)

            # c) Video ZIP
            vid_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"clips_videos/{split}/{video_name}.zip"
            )
            out_vid = os.path.join(args.save_path, "clips_videos", split, f"{video_name}.zip")
            download_and_extract(vid_url, out_vid)

            # d) Audio ZIP
            aud_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"clips_audios/{split}/{video_name}.zip"
            )
            out_aud = os.path.join(args.save_path, "clips_audios", split, f"{video_name}.zip")
            download_and_extract(aud_url, out_aud)

            # e) CSV download + read
            csv_url = (
                f"https://huggingface.co/datasets/"
                f"plnguyen2908/UniTalk-ASD/blob/main/"
                f"csv/{split}/{video_name}.csv"
            )
            out_csv = os.path.join(args.save_path, "csv", f"{split}_{video_name}.csv")
            raw_csv = csv_url.replace("/blob/", "/resolve/")
            subprocess.run(["curl", "-L", "-o", out_csv, raw_csv], check=True)

            # push to list before writing into a big file
            df = pd.read_csv(out_csv, header=None, names=columns)
            df_list.append(df)

            os.remove(out_csv)

    # Merge & write {split}_orig.csv
    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        out_merged = os.path.join(args.save_path, "csv", f"{split}_orig.csv")
        merged.to_csv(out_merged, index=False)
        print(f"Wrote merged CSV for {split}: {out_merged}")

end = time.time()
print(f"The whole script runs in {end - start} seconds")