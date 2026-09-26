import os
import urllib.request
import tarfile
from datetime import datetime
import logging

class AvaDataset:
    """Dataset loader for AVA active speaker detection dataset."""
    
    def __init__(self,
                 root_dir="ava_data",
                 file_list_url="https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt",
                 video_url_template="https://s3.amazonaws.com/ava-dataset/trainval/{}",
                 annotations_url="https://research.google.com/ava/download/ava_activespeaker_val_v1.0.tar.bz2",
                 target_fps=25,resize=None, log_level=logging.INFO, log_dir="logs"):
        """Initialize AVA dataset loader."""
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, "videos")
        self.csv_dir = os.path.join(root_dir, "annotations")
        self.log_dir = log_dir
        self.target_fps = target_fps
        self.resize = resize

        # Track whether directories existed before ensuring them so we can
        # notify on creation vs reuse.
        video_dir_exists = os.path.isdir(self.video_dir)
        csv_dir_exists = os.path.isdir(self.csv_dir)
        log_dir_exists = os.path.isdir(self.log_dir)

        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_list_path = os.path.join(self.csv_dir, "ava_speech_file_names_v1.txt")

        self.file_list_url = file_list_url
        self.video_url_template = video_url_template
        self.annotations_url = annotations_url
        self.logger = logging.getLogger("AvaDataset")
        
        if not self.logger.hasHandlers():
            # Detailed formatter for log file
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # File handler for detailed information
            log_file = os.path.join(
                self.log_dir,
                f"ava_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)

            # Console handler for high-level notifications (downloads, folders, failures)
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.setLevel(log_level)

        # Notify about directory creation or reuse (printed via console handler).
        if not video_dir_exists:
            self.logger.info(f"Created video directory: {self.video_dir}")
        else:
            self.logger.info(f"Using existing video directory: {self.video_dir}")

        if not csv_dir_exists:
            self.logger.info(f"Created annotations directory: {self.csv_dir}")
        else:
            self.logger.info(f"Using existing annotations directory: {self.csv_dir}")

        if not log_dir_exists:
            self.logger.info(f"Created log directory: {self.log_dir}")
        else:
            self.logger.info(f"Using existing log directory: {self.log_dir}")
        
        self._download_annotations()
        self.file_names = self._load_file_list()
        
        self.download_all_videos()

    def _download_annotations(self):
        """Download and extract annotation CSV files if directory is empty."""
        csv_files = [f for f in os.listdir(self.csv_dir) if f.endswith('.csv') and os.path.isfile(os.path.join(self.csv_dir, f))]
        
        if len(csv_files) == 0:
            self.logger.info("Downloading annotation CSV files...")
            tar_path = os.path.join(self.root_dir, "ava_activespeaker_val_v1.0.tar.bz2")
            urllib.request.urlretrieve(self.annotations_url, tar_path)
            with tarfile.open(tar_path, 'r:bz2') as tar:
                tar.extractall(self.csv_dir)
            
            # Move CSV files from subdirectories to csv_dir directly
            for root, dirs, files in os.walk(self.csv_dir):
                if root != self.csv_dir:
                    for file in files:
                        if file.endswith('.csv'):
                            src = os.path.join(root, file)
                            dst = os.path.join(self.csv_dir, file)
                            os.rename(src, dst)
                    # Remove empty subdirectories
                    if not os.listdir(root):
                        os.rmdir(root)
            
            os.remove(tar_path)
            self.logger.info("Annotation CSV files extracted.")
        else:
            self.logger.info("Annotation CSV files already exist.")

    def _load_file_list(self):
        """Load video file names from file list."""
        if not os.path.exists(self.file_list_path):
            self.logger.info("Downloading AVA file list...")
            urllib.request.urlretrieve(self.file_list_url, self.file_list_path)
        with open(self.file_list_path, "r") as f:
            file_names = [line.strip() for line in f if line.strip()]
        self.logger.info(f"Loaded {len(file_names)} video file names.")
        return file_names

    def _download_video(self, file_name):
        """Download video file if not already present."""
        local_path = os.path.join(self.video_dir, file_name)
        if os.path.exists(local_path):
            self.logger.info(f"Video already exists: {file_name}")
            return local_path
        url = self.video_url_template.format(file_name)
        self.logger.info(f"Downloading video: {file_name}")
        urllib.request.urlretrieve(url, local_path)
        return local_path

    def _load_annotation_csv(self, video_name):
        """Load annotation CSV for a video."""
        csv_path = os.path.join(self.csv_dir, f"{video_name}-activespeaker.csv")
        if not os.path.exists(csv_path):
            self.logger.warning(f"CSV annotations not found for video {csv_path}")
            return []
        annots = []
        with open(csv_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                annots.append({
                    "timestamp": float(parts[1]),
                    "bbox": tuple(map(float, parts[2:6])),
                    "label": parts[6],
                    "entity_id": parts[7]
                })
        return annots


    def __len__(self):
        """Return number of videos in dataset."""
        return len(self.file_names)
    
    def download_all_videos(self):
        """Download all videos in the dataset if video_dir is empty."""
        video_files = [f for f in os.listdir(self.video_dir) if os.path.isfile(os.path.join(self.video_dir, f))]
        
        if len(video_files) == 0:
            for file_name in self.file_names:
                self._download_video(file_name)
        else:
            self.logger.info("Video directory is not empty. Skipping download.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download AVA Dataset Videos")
    parser.add_argument("--root_dir", type=str, default="ava_data", help="Root directory for AVA dataset")
    args = parser.parse_args()
    dataset = AvaDataset(root_dir=args.root_dir)
