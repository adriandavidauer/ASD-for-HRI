'''
script to annotate video with bounding boxes, scores and label
'''

# System imports
import argparse

import cv2

# 3rd party imports
from asd4hri.asd import ASD
from asd4hri.asd import load_vvad_classifier
# local imports

# end file header
__author__      = 'Adrian Auer'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video with bounding boxes, scores and label')
    parser.add_argument('--video', help='Path to the video file')
    parser.add_argument('--output_video', help='Path to the output video file')
    parser.add_argument('--architecture', default='CNN2Plus1D_Light', help='Architecture used for the model')
    parser.add_argument('--stride', type=int, default=2, help='Stride for frame sampling')
    parser.add_argument('--averaging_window_size', type=int, default=3, help='Averaging window size for predictions')
    parser.add_argument('--decision_threshold', type=float, default=0.5, help='Decision threshold for predictions')
    parser.add_argument('--weighted', type=bool, default=True, help='Whether to use weighted predictions')
    parser.add_argument('--max_consecutive_empty', type=int, default=2, help='Maximum number of consecutive empty frames before stopping annotation')
    
    args = parser.parse_args()


    pipeline = ASD(architecture=args.architecture, stride=args.stride, averaging_window_size=args.averaging_window_size, decision_threshold=args.decision_threshold, weighted=args.weighted, max_consecutive_empty=args.max_consecutive_empty)
    # # input size to know how many frames we need to go back to set the prediction to the older frames
    # _, input_size = load_vvad_classifier(args.architecture)
    unlabeld_frames = []

    # open video file
    cap = cv2.VideoCapture(args.video)
    # get video extension
    video_extension = args.video.split('.')[-1]
    # get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))


    # iterate over frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # annotate frame with bounding boxes, scores and label
        
