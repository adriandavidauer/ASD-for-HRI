FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
## Dependencies for dlib
RUN apt update && apt install -y git-all build-essential cmake
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/

## Addinitional Deps for OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
## TODO: mount volume for input video and output results
## TODO: adjust command for correct script and arguments
CMD ["python", "src/run_vvad_on_ava_video.py"]