'''
script to annotate video with bounding boxes, scores and label
'''

# System imports
import argparse
import time
import cv2

# 3rd party imports
from asd4hri.asd import ASD, FaceDetector_Options
# from asd4hri.asd import load_vvad_classifier
# from paz.backend.camera import VideoPlayer
from paz.backend.image import resize_image, convert_color_space, show_image, BGR2RGB

# local imports

# end file header
__author__      = 'Adrian Auer'

def fourcc_to_str(fourcc_int):
    return fourcc_int.to_bytes(4, byteorder='little').decode('ascii', errors='ignore')

# TODO: check if it actually works with show False...seems like not the case
def record_from_file(pipeline, video_file_path, name='video.avi',
                         fps=20, fourCC='XVID', image_size=(640, 480), show=False):
        """Adjusted from PAZ, because the version in PAZ does not convert the colors correctly.
        (show_image converts the image to BGR but not back)
        
        Load video and records continuous inference using ``pipeline``.

        # Arguments
            video_file_path: String. Path to the video file.
            name: String. Output video name. Must include the postfix .avi.
            fps: Int. Frames per second.
            fourCC: String. Indicates the four character code of the video.
            e.g. XVID, MJPG, X264.
        """

        fourCC = cv2.VideoWriter_fourcc(*fourCC)
        writer = cv2.VideoWriter(name, fourCC, fps, image_size)

        video = cv2.VideoCapture(video_file_path)
        if (video.isOpened() is False):
            print("Error opening video  file")

        while video.isOpened():
            is_frame_received, frame = video.read()
            if not is_frame_received:
                print("Frame not received. Exiting ...")
                break
            if is_frame_received is True:
                frame = convert_color_space(frame, BGR2RGB)
                output = pipeline(frame)
                if output is None:
                    continue
                image = resize_image(output['image'], tuple(image_size))
                if show:
                    show_image(image, 'inference', wait=False)
                    image = convert_color_space(image, BGR2RGB)
                writer.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotate video with bounding boxes, scores and label')
    parser.add_argument('--video', help='Path to the video file')
    parser.add_argument('--output_video', help='Path to the output video file')
    parser.add_argument('--show', help='flag to show the video or not', action="store_true")
    parser.add_argument('--architecture', default='CNN2Plus1D_Light', help='Architecture used for the model')
    parser.add_argument('--stride', type=int, default=1, help='Stride for frame sampling')
    parser.add_argument('--averaging_window_size', type=int, default=3, help='Averaging window size for predictions')
    parser.add_argument('--decision_threshold', type=float, default=0.5, help='Decision threshold for predictions')
    parser.add_argument('--weighted', type=bool, default=True, help='Whether to use weighted predictions')
    parser.add_argument('--max_consecutive_empty', type=int, default=2, help='Maximum number of consecutive empty frames before stopping annotation')
    parser.add_argument('--face_detector', type=str, default='YuNet', choices=FaceDetector_Options, help='the face detector for the ASD pipeline')
    args = parser.parse_args()


    pipeline = ASD(architecture=args.architecture, stride=args.stride, averaging_window_size=args.averaging_window_size, decision_threshold=args.decision_threshold, weighted=args.weighted, max_consecutive_empty=args.max_consecutive_empty, annotate_output=True, detector=args.face_detector) 

    # open video file
    cap = cv2.VideoCapture(args.video)
    # get video extension
    video_extension = args.video.split('.')[-1]
    # get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    t_0 = time.time()
    # player = VideoPlayer((width, height), pipeline, None)
    record_from_file(pipeline=pipeline, video_file_path=args.video, name=args.output_video, fps=fps, fourCC=fourcc_to_str(fourcc), image_size=(width, height), show=args.show)
    print(f'saved results to {args.output_video} in {time.time() - t_0} seconds')




        
