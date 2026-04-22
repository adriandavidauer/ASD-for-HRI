import os
import argparse
import cv2
import numpy as np
import logging
import time
from collections import defaultdict
from datetime import datetime

from ava_dataset import AvaDataset
import paz.pipelines.detection as dt
from paz.backend.boxes import compute_iou as backend_compute_iou


LOGGER = logging.getLogger('VVAD_Evaluation')


def setup_logging(log_file_path=None, enable_logging=True):
    """Set up logging for VVAD evaluation.

    # Arguments
        log_file_path: String or None - Optional path to a log file. If ``None``,
            a timestamped log file is created under a ``logs`` directory in the
            current working directory.
        enable_logging: Boolean - If ``True``, attaches both a file handler and
            a console handler to the logger. If ``False``, only a console
            handler is attached (no log file is created).

    Returns:
        logging.Logger: Configured logger named ``'VVAD_Evaluation'``.
    """
    # Configure shared module logger
    LOGGER.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    LOGGER.handlers.clear()
    
    # Console handler for user-facing notifications (downloads, errors, summaries)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    LOGGER.addHandler(console_handler)

    # Optional file handler for detailed logs
    if enable_logging:
        if log_file_path is None:
            # Create logs directory if it doesn't exist
            logs_dir = 'logs'
            os.makedirs(logs_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(logs_dir, f"vvad_evaluation_{timestamp}.log")

        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)

        detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(detailed_formatter)

        LOGGER.addHandler(file_handler)
    
    return LOGGER

def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments for running the VVAD
        evaluation script.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate DetectVVAD on an AVA video with multiple faces')
    parser.add_argument('--root_dir', type=str,
                        default=os.path.expanduser('~/PAZ/ava-data'),
                        help='Root directory of AVA data (with videos/ and annotations/)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to a specific AVA video file to process')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (default: <video_basename>_vvad.<ext>)')
    parser.add_argument('--output_ext', type=str, default='mp4',
                        help='Output video file extension (default: mp4). Options: mp4, avi, mov, mkv')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for matching predicted boxes to GT boxes')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Log file path (default: logs/vvad_evaluation_TIMESTAMP.log)')
    parser.add_argument('--enable_logging', action='store_true',
                        help='Enable logging (default: True)')
    return parser.parse_args()


def pick_first_annotated_video(dataset: AvaDataset):
    """Find the first video in the dataset that has annotations.

    If the user does not specify a video explicitly, this function searches
    the dataset for the first video that has a non-empty annotation CSV.

    # Arguments
        dataset: AvaDataset - Initialized dataset instance providing file names
            and paths to videos and annotation CSVs.

    Returns:
        str | None: Absolute path to the first annotated video if one is found,
        otherwise ``None``.
    """
    for file_name in dataset.file_names:
        video_name = os.path.splitext(file_name)[0]
        csv_path = os.path.join(dataset.csv_dir, f"{video_name}-activespeaker.csv")
        if not os.path.exists(csv_path):
            continue
        annots = dataset._load_annotation_csv(video_name)
        if annots != []:
            return os.path.join(dataset.video_dir, file_name)
    return None


def _map_ava_to_vvad_text(ava_numeric_label: int) -> str:
    """Map numeric AVA labels to VVAD text labels.

    The mapping collapses both speaking states into a single
    ``'speaking'`` label to match the VVAD output format.

    # Arguments
        ava_numeric_label: Integer - AVA label where
            ``0`` = not speaking,
            ``1`` = speaking but not audible,
            ``2`` = speaking and audible.

    Returns:
        str: VVAD-style label, either ``'speaking'``, ``'not-speaking'``, or
        ``'error'`` if the input label is invalid.
    """
    if ava_numeric_label == 0:
        return 'not-speaking'
    elif ava_numeric_label == 1:
        return 'speaking'
    elif ava_numeric_label == 2:
        return 'speaking'
    LOGGER.error(f"Invalid AVA label: {ava_numeric_label}")
    return 'error'


def compute_iou(box_a, box_b):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    This is a thin adapter around :func:`paz.backend.boxes.compute_iou`
    that preserves the two-box interface used in this script.

    # Arguments
        box_a: List[float] - Iterable of floats - Four numeric values
            ``[x_min, y_min, x_max, y_max]`` for the first box.
        box_b: List[float] - Iterable of floats - Four numeric values
            ``[x_min, y_min, x_max, y_max]`` for the second box.

    Returns:
        float: IoU value between ``0.0`` and ``1.0``.
    """
    box = np.array(box_a, dtype=np.float32)
    boxes = np.array([box_b], dtype=np.float32)
    ious = backend_compute_iou(box, boxes)
    return float(ious[0])


def _compute_intersection_and_areas(box_a, box_b):
    """Compute intersection area and individual areas for two boxes.

    # Arguments
        box_a: Sequence[float] - ``[x_min, y_min, x_max, y_max]``.
        box_b: Sequence[float] - ``[x_min, y_min, x_max, y_max]``.

    Returns:
        tuple[float, float, float]: ``(intersection_area, area_a, area_b)``.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))

    return intersection_area, area_a, area_b


def match_boxes_to_gt(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes using IoU.

    # Arguments
        pred_boxes: Sequence - Predicted box objects. Each object is expected
            to expose a ``coordinates`` attribute with
            ``[x_min, y_min, x_max, y_max]`` in pixel space.
        gt_boxes: Sequence - Ground-truth annotation dictionaries, each
            containing at least a ``'bbox_pixel'`` key with
            ``[x_min, y_min, x_max, y_max]`` coordinates.
        iou_threshold: Float - Minimum IoU required for a prediction–GT pair to
            be considered a valid match based on standard IoU (intersection
            over union). Additionally, pairs can be accepted when one box is
            (almost) fully contained in the other, even if IoU is below this
            threshold.

    Returns:
        list[tuple[int, int, float]]: A list of matched triplets
        ``(pred_idx, gt_idx, iou)`` where ``pred_idx`` is the index of the
        prediction, ``gt_idx`` is the index of the matched ground-truth box,
        and ``iou`` is the IoU score for the pair.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        LOGGER.debug(
            "No matching possible: pred_boxes=%d, gt_boxes=%d",
            len(pred_boxes),
            len(gt_boxes),
        )
        return []
    
    LOGGER.debug(
        "Matching %d predictions against %d GT boxes (IoU ≥ %.2f or high containment)",
        len(pred_boxes),
        len(gt_boxes),
        iou_threshold,
    )
    
    matches = []
    used_gt_indices = set()
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        match = _find_best_match(
            pred_idx,
            pred_box,
            gt_boxes,
            used_gt_indices,
            iou_threshold,
        )
        if match:
            matches.append(match)
            used_gt_indices.add(match[1])
            LOGGER.debug(
                "Matched prediction %d → GT %d (IoU=%.3f)",
                pred_idx,
                match[1],
                match[2],
            )
        else:
            LOGGER.debug(
                "No match for prediction %d (best IoU/containment below thresholds; IoU ≥ %.2f)",
                pred_idx,
                iou_threshold,
            )
    
    return matches


def _find_best_match(pred_idx, pred_box, gt_boxes, used_gt_indices, iou_threshold):
    """Find the best unused ground-truth match for a prediction.

    # Arguments
        pred_idx: Integer - Index of the predicted box in the prediction list.
        pred_box: Object - Prediction object with a ``coordinates`` attribute
            describing the bounding box in pixel coordinates.
        gt_boxes: Sequence - Ground-truth dictionaries, each containing
            ``'bbox_pixel'`` coordinates.
        used_gt_indices: Set of integers - Ground-truth indices that have
            already been matched and should not be matched again.
        iou_threshold: Float - Minimum IoU required for a prediction–GT pair to
            be accepted as a match based on standard IoU. Additionally, a pair
            can be accepted when one box is almost fully contained within the
            other (high containment), even if IoU is below this threshold.

    Returns:
        tuple[int, int, float] | None: Triplet ``(pred_idx, gt_idx, iou)`` for
        the best match that meets either the IoU or containment thresholds, or
        ``None`` if no match satisfies the thresholds.
    """
    CONTAINMENT_THRESHOLD = 0.8  # require at least 80% containment of the smaller box

    pred_coords = pred_box.coordinates
    best_iou = 0.0
    best_containment = 0.0
    best_gt_idx = None
    
    for gt_idx, gt_box in enumerate(gt_boxes):
        if gt_idx in used_gt_indices:
            continue
        
        gt_coords = gt_box['bbox_pixel']
        iou = compute_iou(pred_coords, gt_coords)

        # Compute how much of the smaller box is actually overlapped
        inter_area, area_pred, area_gt = _compute_intersection_and_areas(pred_coords, gt_coords)
        min_area = max(1e-6, min(area_pred, area_gt))
        containment = inter_area / min_area

        LOGGER.debug(
            "Candidate match: Pred %d vs GT %d → IoU=%.3f, containment=%.3f",
            pred_idx,
            gt_idx,
            iou,
            containment,
        )

        # Decide whether this candidate is better than the current best.
        # We prioritise higher IoU, but also consider containment as a tie-breaker
        # so that boxes that are almost entirely inside each other are preferred.
        if (iou > best_iou) or (iou == best_iou and containment > best_containment):
            best_iou = iou
            best_containment = containment
            best_gt_idx = gt_idx
    
    # Accept a match if either:
    # - IoU is above the configured threshold, OR
    # - the smaller box is at least CONTAINMENT_THRESHOLD covered by the other box.
    if best_gt_idx is not None and (
        best_iou >= iou_threshold or best_containment >= CONTAINMENT_THRESHOLD
    ):
        LOGGER.debug(
            "Best match for pred %d: GT %d (IoU=%.3f, containment=%.3f, threshold=%.3f)",
            pred_idx,
            best_gt_idx,
            best_iou,
            best_containment,
            iou_threshold,
        )
        return (pred_idx, best_gt_idx, best_iou)

    LOGGER.debug(
        "No acceptable match for pred %d (best IoU=%.3f, best containment=%.3f, IoU threshold=%.3f, containment threshold=%.3f)",
        pred_idx,
        best_iou,
        best_containment,
        iou_threshold,
        CONTAINMENT_THRESHOLD,
    )
    return None


def initialize_video_capture(video_path: str):
    """Initialize a video capture object for the given video file.

    # Arguments
        video_path: String - Path to the input video file.

    Returns:
        cv2.VideoCapture: OpenCV video capture object positioned at the first
        frame of the video.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video file: {video_path}')
    return cap


def get_video_properties(cap, dataset: AvaDataset):
    """Extract video properties from a capture object and validate the first frame.

    # Arguments
        cap: cv2.VideoCapture - OpenCV capture instance for the video.
        dataset: AvaDataset - Dataset providing a fallback ``target_fps`` if
            the video's FPS cannot be read.

    Returns:
        tuple[float, int, int]: A tuple ``(fps, width, height)`` where
        ``fps`` is the effective frames per second, and ``width``/``height``
        are the frame dimensions in pixels.

    Raises:
        ValueError: If the first frame cannot be read or the dimensions are
            invalid.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or dataset.target_fps
    if fps <= 0:
        fps = dataset.target_fps
    
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise ValueError('Could not read first frame')
    
    height, width = first_frame.shape[:2]
    if height <= 0 or width <= 0:
        cap.release()
        raise ValueError(f'Invalid frame dimensions: {width}x{height}')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return fps, width, height


def get_evaluation_timestamps(annots, min_frames=38, stride=2):
    """Derive timestamps at which VVAD predictions should be evaluated.

    # Arguments
        annots: Sequence - Annotation dictionaries, each expected to contain
            a ``'timestamp'`` key in seconds.
        min_frames: Integer - Minimum number of timestamps to skip initially to
            allow the VVAD buffer to warm up. Currently not applied to the
            returned list but retained for configuration and logging.
        stride: Integer - Intended stride between evaluation timestamps.
            Currently not applied to the returned list but retained for
            configuration and potential future use.

    Returns:
        list[float]: Sorted list of unique timestamps (in seconds) extracted
        from the annotations.
    """
    if not annots:
        return []
    
    timestamps = sorted(set(annot['timestamp'] for annot in annots))
    if not timestamps:
        return []
    
    LOGGER.info('All available timestamps: %d total', len(timestamps))
    LOGGER.info(
        'Timestamp range: %.2f to %.2f seconds', timestamps[0], timestamps[-1]
    )
    LOGGER.debug('Sample timestamps: %s', timestamps[:10])
    
    # Start from min_frames to allow VVAD buffer to mature
    start_idx = min(min_frames, len(timestamps))
    eval_timestamps = timestamps  # timestamps[start_idx::stride]
    LOGGER.info(
        'After applying min_frames=%d and stride=%d: %d timestamps',
        min_frames,
        stride,
        len(eval_timestamps),
    )
    LOGGER.debug('Evaluation timestamps: %s', eval_timestamps[:10])
    
    return eval_timestamps


def timestamp_to_frame_index(timestamp, fps):
    """Convert a timestamp in seconds to a zero-based frame index.

    # Arguments
        timestamp: Float - Time in seconds.
        fps: Float - Frames per second of the video.

    Returns:
        int: Closest integer frame index corresponding to the timestamp.
    """
    return int(round(timestamp * fps))


def map_annotations_to_frames(annots, fps, dataset: AvaDataset):
    """Map annotations to frame indices based on their timestamps.

    # Arguments
        annots: Sequence - Annotation dictionaries with at least ``'timestamp'``,
            ``'bbox'``, ``'label'``, and optionally ``'entity_id'`` keys.
        fps: Float - Frames per second of the video.
        dataset: AvaDataset - Dataset instance. Currently unused but kept for
            interface consistency.

    Returns:
        tuple[dict[int, list[dict]], float, float]: A tuple containing:
            - A dictionary mapping frame indices to lists of annotation
              dictionaries enriched with ``'bbox_normalized'``, ``'label'``,
              ``'entity_id'``, and ``'timestamp'``.
            - A dummy start time value (``0.0``).
            - The frame duration in seconds (``1.0 / fps``).
    """
    timestamp_to_annotations = {}
    
    for annot in annots:
        timestamp = annot.get('timestamp', 0)
        frame_idx = timestamp_to_frame_index(timestamp, fps)
        
        if frame_idx not in timestamp_to_annotations:
            timestamp_to_annotations[frame_idx] = []
        
        timestamp_to_annotations[frame_idx].append({
            'bbox_normalized': annot['bbox'],
            'label': annot['label'],
            'entity_id': annot.get('entity_id', ''),
            'timestamp': timestamp
        })
    
    return timestamp_to_annotations, 0.0, 1.0/fps


def convert_gt_boxes_to_pixel(gt_boxes_for_frame, width, height, dataset: AvaDataset):
    """Convert normalized ground-truth bounding boxes to pixel coordinates.

    # Arguments
        gt_boxes_for_frame: List[dict] - Annotation dictionaries for
            a single frame, each containing normalized ``'bbox_normalized'``
            coordinates and associated metadata.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        dataset: AvaDataset - Dataset instance used to map AVA labels to VVAD
            labels.

    Returns:
        list[dict]: List of dictionaries with keys
        ``'bbox_pixel'``, ``'label'``, ``'entity_id'``, and ``'gt_vvad_label'``,
        where ``'bbox_pixel'`` is in pixel coordinates.
    """
    gt_boxes_pixel = []
    for gt_ann in gt_boxes_for_frame:
        gt_box_pixel = _convert_single_box_to_pixel(gt_ann, width, height, dataset)
        gt_boxes_pixel.append(gt_box_pixel)
    return gt_boxes_pixel


def _convert_single_box_to_pixel(gt_ann, width, height, dataset: AvaDataset):
    """Convert a single normalized annotation box to pixel coordinates.

    # Arguments
        gt_ann: Dictionary - Annotation dictionary containing a
            ``'bbox_normalized'`` field and label information.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        dataset: AvaDataset - Dataset instance used to map the AVA CSV label to
            a numeric label and then to a VVAD-style text label.

    Returns:
        dict: Dictionary with pixel-space bounding box and associated metadata,
        including keys ``'bbox_pixel'``, ``'label'``, ``'entity_id'``, and
        ``'gt_vvad_label'``.
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = gt_ann['bbox_normalized']
    bbox_pixel = (
        x_min_norm * width, y_min_norm * height,
        x_max_norm * width, y_max_norm * height
    )
    
    # Convert label using dataset mapping
    csv_label = gt_ann['label']
    numeric_label = dataset._map_label(csv_label)
    gt_vvad_label = _map_ava_to_vvad_text(numeric_label)
    
    # Debug: print coordinate conversion
    LOGGER.debug("Converting GT box for entity %s", gt_ann['entity_id'])
    LOGGER.debug("  Normalized: %s", gt_ann['bbox_normalized'])
    LOGGER.debug("  Pixel: %s", (bbox_pixel,))
    LOGGER.debug("  Image size: %dx%d", width, height)
    LOGGER.debug(
        "  Label: %s -> %s (numeric: %s)", csv_label, gt_vvad_label, numeric_label
    )
    
    return {
        'bbox_pixel': bbox_pixel,
        'label': gt_ann['label'],
        'entity_id': gt_ann['entity_id'],
        'gt_vvad_label': gt_vvad_label
    }


def process_frame_predictions(frame_rgb, pipeline):
    """Run a frame through the VVAD detection pipeline.

    # Arguments
        frame_rgb: numpy.ndarray - Input frame as an RGB image.
        pipeline: Callable - VVAD detection pipeline (e.g. ``dt.DetectVVAD``)
            that accepts an RGB image and returns a dictionary with at least an
            ``'image'`` key and optionally a ``'boxes2D'`` key.

    Returns:
        tuple[numpy.ndarray, list]: A tuple ``(out_rgb, pred_boxes)`` where
        ``out_rgb`` is the (possibly modified) RGB frame returned by the
        pipeline and ``pred_boxes`` is the list of predicted box objects.
    """
    result = pipeline(frame_rgb)
    out_rgb = result['image'] if isinstance(result, dict) and 'image' in result else frame_rgb
    pred_boxes = result.get('boxes2D', []) if isinstance(result, dict) else []
    
    # Debug: print prediction details and coordinate format
    LOGGER.debug("Pipeline predictions:")
    LOGGER.debug("  Result type: %s", type(result))
    LOGGER.debug("  Number of predicted boxes: %d", len(pred_boxes))
    LOGGER.debug("  Frame shape: %s", frame_rgb.shape)
    
    for i, pred_box in enumerate(pred_boxes):
        coords = getattr(pred_box, 'coordinates', None)
        class_name = getattr(pred_box, 'class_name', 'None')
        score = getattr(pred_box, 'score', 'None')
        
        LOGGER.debug("  Pred %d: class_name=%s, score=%s", i, class_name, score)
        LOGGER.debug("        coordinates=%s", coords)
        
        # Check if coordinates are normalized (0-1) or pixel coordinates
        if coords is not None and len(coords) >= 4:
            x_min, y_min, x_max, y_max = coords[:4]
            if 0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1:
                LOGGER.debug(
                    "        → Coordinates appear to be NORMALIZED (0-1 range)"
                )
            elif x_max > 1 or y_max > 1:
                LOGGER.debug(
                    "        → Coordinates appear to be PIXEL coordinates "
                    "(max: %.1f, %.1f)",
                    x_max,
                    y_max,
                )
            else:
                LOGGER.debug("        → Coordinate format unclear")
    
    return out_rgb, pred_boxes


def draw_prediction_overlay(out_rgb, pred_box, gt_label, iou, width, height):
    """Draw a prediction overlay (label, status, IoU) on a frame.

    # Arguments
        out_rgb: numpy.ndarray - RGB frame image to be modified in place.
        pred_box: Object - Prediction object with ``coordinates`` and
            ``class_name`` attributes.
        gt_label: String - Ground-truth VVAD-style label for the matched entity
            (e.g. ``'speaking'`` or ``'not-speaking'``).
        iou: Float - IoU score between the predicted box and ground-truth box.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.

    Returns:
        None
    """
    pred_label = getattr(pred_box, 'class_name', None)
    if pred_label is None:
        return
    
    status, color = _get_prediction_status(pred_label, gt_label)
    text_x, text_y = _get_text_position(pred_box, width, height)
    text = f"{pred_label} | {status} (IoU:{iou:.2%})"  # Show IoU as percentage
    
    try:
        cv2.putText(
            out_rgb,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,  # smaller font size for less clutter
            color,
            1,
            cv2.LINE_AA,
        )
    except Exception as e:
        LOGGER.warning('Failed to draw text: %s', e)


def _get_prediction_status(pred_label, gt_label):
    """Determine prediction correctness and visualization color.

    # Arguments
        pred_label: String - Predicted VVAD label.
        gt_label: String - Ground-truth VVAD label.

    Returns:
        tuple[str, tuple[int, int, int]]: Status string (``'OK'`` or
        ``'WRONG'``) and BGR color tuple suitable for ``cv2.putText``.
    """
    status = 'OK' if pred_label == gt_label else 'WRONG'
    color = (0, 255, 0) if pred_label == gt_label else (255, 0, 0)
    return status, color


def _get_text_position(pred_box, width, height):
    """Calculate a suitable text position for a prediction overlay.

    # Arguments
        pred_box: Object - Prediction object with ``coordinates`` attribute.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.

    Returns:
        tuple[int, int]: ``(x, y)`` coordinates for drawing overlay text.
    """
    x_min, y_min, x_max, y_max = pred_box.coordinates
    text_x = max(0, min(int(x_min), width - 200))
    text_y = max(20, min(int(y_min) - 10, height - 10))
    return text_x, text_y


def draw_frame_summary(out_rgb, frame_idx, frame_correct, frame_matched, height):
    """Draw a frame-level accuracy summary overlay.

    # Arguments
        out_rgb: numpy.ndarray - RGB frame image to be modified in place.
        frame_idx: Integer - Index of the current frame.
        frame_correct: Integer - Number of correct predictions in the current
            frame.
        frame_matched: Integer - Number of matched prediction–GT pairs in the
            frame.
        height: Integer - Frame height in pixels.

    Returns:
        None
    """
    try:
        if frame_matched > 0:
            frame_accuracy = frame_correct / frame_matched
            text = f"Frame {frame_idx}: {frame_correct}/{frame_matched} correct ({frame_accuracy:.2f})"
            color = (255, 255, 255)
        else:
            text = f"Frame {frame_idx}: No matches"
            color = (0, 255, 255)
        
        cv2.putText(
            out_rgb,
            text,
            (10, max(25, height - 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,  # smaller font size
            color,
            1,
            cv2.LINE_AA,
        )
    except Exception as e:
        LOGGER.warning('Failed to draw frame summary: %s', e)


def draw_gt_boxes_overlay(out_rgb, gt_boxes_pixel, width, height):
    """Draw all ground-truth boxes for a frame for visual inspection.

    GT boxes are drawn independently of matching so that overlap between
    predicted and ground-truth regions can be inspected visually.

    # Arguments
        out_rgb: numpy.ndarray - RGB frame image to be modified in place.
        gt_boxes_pixel: List[dict] - Ground-truth boxes in pixel coordinates,
            each containing ``'bbox_pixel'``, ``'entity_id'``, and
            ``'gt_vvad_label'``.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.

    Returns:
        None
    """
    try:
        for gt_box in gt_boxes_pixel:
            x_min, y_min, x_max, y_max = gt_box['bbox_pixel']
            entity_id = gt_box.get('entity_id', '')
            gt_label = gt_box.get('gt_vvad_label', '')

            # Clamp coordinates to image bounds
            x_min_i = max(0, min(int(x_min), width - 1))
            y_min_i = max(0, min(int(y_min), height - 1))
            x_max_i = max(0, min(int(x_max), width - 1))
            y_max_i = max(0, min(int(y_max), height - 1))

            # Yellow rectangle for GT boxes
            cv2.rectangle(out_rgb, (x_min_i, y_min_i), (x_max_i, y_max_i), (0, 255, 255), 2)

            text = f"GT {entity_id} {gt_label}"
            text_y = max(8, y_min_i - 4)
            cv2.putText(
                out_rgb,
                text,
                (x_min_i, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,  # smaller font size
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    except Exception as e:
        LOGGER.warning('Failed to draw GT boxes: %s', e)


def initialize_entity_tracking():
    """Initialize dictionaries for tracking per-entity statistics.

    Returns:
        dict: A nested dictionary structure containing counters for predictions,
        correctness, ground-truth labels, and totals per entity.
    """
    return {
        'entity_predictions': defaultdict(int),
        'entity_correct': defaultdict(int),
        'entity_matched': defaultdict(int),
        'entity_labels': defaultdict(lambda: defaultdict(int)),
        'entity_gt_labels': defaultdict(lambda: defaultdict(int)),  # GT labels for matched annotations (mapped to VVAD: 'speaking' or 'not-speaking')
        'entity_gt_total': defaultdict(int),  # Total GT annotations per entity from CSV
        'entity_gt_labels_all': defaultdict(lambda: defaultdict(int))  # All GT labels from CSV (mapped: 'speaking' or 'not-speaking')
    }


def update_entity_stats(entity_tracking, entity_id, pred_label, gt_label, is_correct):
    """Update per-entity statistics given a single prediction result.

    # Arguments
        entity_tracking: Dictionary - Nested dictionary structure created by
            ``initialize_entity_tracking``.
        entity_id: String - Identifier of the entity (e.g. face track ID).
        pred_label: String - Predicted VVAD label for this entity.
        gt_label: String - Ground-truth VVAD label for this entity.
        is_correct: Boolean - Whether the prediction matches the ground-truth
            label.

    Returns:
        None
    """
    entity_tracking['entity_predictions'][entity_id] += 1
    entity_tracking['entity_labels'][entity_id][pred_label] += 1
    entity_tracking['entity_matched'][entity_id] += 1
    
    if is_correct:
        entity_tracking['entity_correct'][entity_id] += 1
    
    # Track ground truth labels for analysis
    if gt_label not in entity_tracking['entity_gt_labels'][entity_id]:
        entity_tracking['entity_gt_labels'][entity_id][gt_label] = 0
    entity_tracking['entity_gt_labels'][entity_id][gt_label] += 1


def setup_evaluation(video_path: str, dataset: AvaDataset):
    """Set up VVAD evaluation components for a given video.

    # Arguments
        video_path: String - Path to the input video file.
        dataset: AvaDataset - Dataset instance used to load annotations for the
            video.

    Returns:
        tuple: ``(pipeline, video_name, annots)`` where
        ``pipeline`` is the VVAD detection pipeline,
        ``video_name`` is the base video name without extension, and
        ``annots`` is the list of loaded annotations.

    Raises:
        ValueError: If no annotations are found for the video.
    """
    pipeline = dt.DetectVVAD()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annots = dataset._load_annotation_csv(video_name)
    
    if annots == []:
        raise ValueError(f'No annotations found for video {video_name}')
    
    return pipeline, video_name, annots


def get_video_codec(output_path: str):
    """Select an appropriate video codec based on the output file extension.

    # Arguments
        output_path: String - Target output video path including extension.

    Returns:
        str: Four-character code string (e.g. ``'mp4v'``, ``'XVID'``) to be
        used with ``cv2.VideoWriter_fourcc``.
    """
    _, ext = os.path.splitext(output_path.lower())
    
    codec_map = {
        '.mp4': 'mp4v',  # MPEG-4 codec
        '.avi': 'XVID',  # Xvid codec
        '.mov': 'mp4v',  # QuickTime compatible
        '.mkv': 'X264',  # H.264 codec
    }
    
    codec = codec_map.get(ext, 'mp4v')  # Default to mp4v
    return codec


def setup_video_io(video_path: str, output_path: str, dataset: AvaDataset):
    """Initialize video capture and writer for evaluation output.

    # Arguments
        video_path: String - Path to the input video file.
        output_path: String - Path where the annotated output video will be
            saved.
        dataset: AvaDataset - Dataset instance providing the target FPS.

    Returns:
        tuple: ``(cap, writer, fps, width, height)`` where ``cap`` is the
        input ``VideoCapture``, ``writer`` is the output ``VideoWriter``,
        ``fps`` is the effective frames per second used for evaluation, and
        ``width``/``height`` are frame dimensions.

    Raises:
        RuntimeError: If the video writer cannot be opened with the selected
            codec and output path.
    """
    cap = initialize_video_capture(video_path)
    fps, width, height = get_video_properties(cap, dataset)
    
    # Get appropriate codec based on output file extension
    codec = get_video_codec(output_path)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, dataset.target_fps, (width, height))
    
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Failed to open video writer for {output_path} with codec {codec}')
    
    return cap, writer, fps, width, height


def initialize_ground_truth_entity_counts(annots, dataset: AvaDataset):
    """Initialize ground-truth entity counts from all CSV annotations.

    Both speaking states (speaking-audible and speaking-not-audible) are
    mapped to ``'speaking'`` to match the model output format, which only
    distinguishes ``'speaking'`` and ``'not-speaking'``.

    # Arguments
        annots: List[dict] - Raw annotation dictionaries loaded from CSV.
        dataset: AvaDataset - Dataset instance used to map AVA labels to
            numeric IDs and then to VVAD-style text labels.

    Returns:
        tuple[defaultdict, defaultdict]: A pair ``(entity_gt_total,
        entity_gt_labels_all)`` where:
            - ``entity_gt_total`` maps ``entity_id`` to the total number of
              annotations in the CSV.
            - ``entity_gt_labels_all`` maps ``entity_id`` to a dictionary of
              VVAD-style labels and their counts.
    """
    entity_gt_total = defaultdict(int)
    entity_gt_labels_all = defaultdict(lambda: defaultdict(int))
    
    for annot in annots:
        entity_id = annot.get('entity_id', '')
        if entity_id:
            entity_gt_total[entity_id] += 1
            label = annot.get('label', '')
            if label:
                # Map to VVAD format: both speaking types -> 'speaking', not-speaking -> 'not-speaking'
                numeric_label = dataset._map_label(label)
                mapped_label = _map_ava_to_vvad_text(numeric_label)
                entity_gt_labels_all[entity_id][mapped_label] += 1
    
    return entity_gt_total, entity_gt_labels_all


def prepare_evaluation_data(annots, fps, dataset: AvaDataset):
    """Prepare timestamps and frame mappings for evaluation.

    # Arguments
        annots: List[dict] - Annotation dictionaries for the video.
        fps: Float - Frames per second of the video.
        dataset: AvaDataset - Dataset instance used for frame/label operations.

    Returns:
        tuple[dict[int, list[dict]], list[int]]: A tuple
        ``(timestamp_to_annotations, eval_frame_indices)`` where
        ``timestamp_to_annotations`` maps frame indices to their annotations
        and ``eval_frame_indices`` is the list of frame indices where
        evaluation should be performed.
    """
    eval_timestamps = get_evaluation_timestamps(annots)
    LOGGER.info('Evaluating at %d timestamps', len(eval_timestamps))
    LOGGER.debug(
        'Evaluation timestamps: %s...', eval_timestamps[:10]
    )  # Show first 10
    
    timestamp_to_annotations, _, _ = map_annotations_to_frames(annots, fps, dataset)
    
    LOGGER.info(
        'Total annotation frames: %d', len(timestamp_to_annotations)
    )
    
    if len(timestamp_to_annotations) > 0:
        frame_range = (
            f"{min(timestamp_to_annotations.keys())} "
            f"to {max(timestamp_to_annotations.keys())}"
        )
        LOGGER.info('Frame indices range: %s', frame_range)
        LOGGER.debug(
            'Sample annotated frames: %s',
            sorted(list(timestamp_to_annotations.keys()))[:10],
        )
    
    eval_frame_indices = [timestamp_to_frame_index(ts, fps) for ts in eval_timestamps]
    LOGGER.debug(
        'Evaluation frame indices: %s...', eval_frame_indices[:10]
    )  # Show first 10
    return timestamp_to_annotations, eval_frame_indices


def initialize_statistics():
    """Initialize global statistics tracking dictionaries for evaluation.

    Returns:
        dict: Top-level dictionary containing counters for matched prediction–
        ground-truth pairs, frame-level prediction counts, correctness, label
        counts, total prediction time, and nested per-entity statistics.
    """
    return {
        # Number of GT–prediction matches over all frames (IoU/containment +
        # label-agnostic spatial matching).
        'total_matched': 0,
        # Number of spatial matches where the VVAD label (speaking/not-speaking)
        # is also correct.
        'total_correct': 0,
        'speaking_count': 0,
        'not_speaking_count': 0,
        # Raw pipeline outputs (before IoU matching), for debugging how often
        # the VVAD pipeline produces boxes.
        'total_pipeline_boxes': 0,
        # Number of frames for which the pipeline produced at least one box
        # (i.e. "total frame predictions").
        'total_frame_predictions': 0,
        # Timestamp-based accuracy: denominator is all GT annotations on
        # evaluation frames (based only on timestamp/frame). Numerator is the
        # number of those GT annotations that have at least one correct VVAD
        # label with sufficient spatial overlap. This numerator is the final
        # count of correct predictions under the timestamp-based protocol.
        'total_eval_gt_annotations': 0,
        'final_correct_predictions': 0,
        'total_prediction_time': 0.0,
        'entity_tracking': initialize_entity_tracking()
    }


def process_single_match(pred_box, gt_box, iou, width, height, stats, out_rgb):
    """Process a single matched prediction–ground-truth pair.

    # Arguments
        pred_box: Object - Prediction object with ``class_name`` and
            ``coordinates``.
        gt_box: Dictionary - Ground-truth dictionary containing at least
            ``'gt_vvad_label'`` and ``'entity_id'`` keys.
        iou: Float - IoU score between the prediction and ground-truth box.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        stats: Dictionary - Global statistics dictionary created by
            ``initialize_statistics``.
        out_rgb: numpy.ndarray - RGB frame image to be modified in place with
            overlays.

    Returns:
        bool: ``True`` if the prediction label matches the ground-truth label,
        otherwise ``False``.
    """
    pred_label = getattr(pred_box, 'class_name', None)
    if pred_label is None:
        return False
    
    stats['speaking_count'] += pred_label == 'speaking'
    stats['not_speaking_count'] += pred_label == 'not-speaking'
    
    gt_label = gt_box['gt_vvad_label']
    is_correct = pred_label == gt_label
    
    if is_correct:
        stats['total_correct'] += 1
    
    entity_id = gt_box['entity_id']
    update_entity_stats(stats['entity_tracking'], entity_id, pred_label, gt_label, is_correct)
    draw_prediction_overlay(out_rgb, pred_box, gt_label, iou, width, height)
    
    return is_correct


def evaluate_frame(frame_count, pred_boxes, timestamp_to_annotations, width, height,
                  dataset: AvaDataset, iou_threshold, stats, out_rgb):
    """Evaluate a single frame against ground-truth annotations.

    # Arguments
        frame_count: Integer - Index of the current frame being evaluated.
        pred_boxes: List[object] - Prediction objects for the frame.
        timestamp_to_annotations: Dict[int, List[dict]] - Mapping from frame
            indices to lists of annotation dictionaries.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        dataset: AvaDataset - Dataset instance used for label mapping.
        iou_threshold: Float - Minimum IoU required to match a prediction to a
            GT box.
        stats: Dictionary - Global statistics dictionary being accumulated over
            frames.
        out_rgb: numpy.ndarray - RGB frame image to be modified in place with
            overlays.

    Returns:
        int: Number of correct predictions for the current frame.
    """
    gt_boxes_for_frame = timestamp_to_annotations.get(frame_count, [])
    gt_boxes_pixel = convert_gt_boxes_to_pixel(gt_boxes_for_frame, width, height, dataset)
    
    LOGGER.debug('=== Frame %d Evaluation ===', frame_count)
    LOGGER.debug('Predicted boxes: %d', len(pred_boxes))
    LOGGER.debug('Ground truth boxes: %d', len(gt_boxes_for_frame))
    
    # Debug: Check if this frame should have annotations
    if frame_count not in timestamp_to_annotations:
        LOGGER.debug(
            'Frame %d not found in timestamp_to_annotations!', frame_count
        )
        LOGGER.debug(
            'Available frames with annotations: %s...',
            sorted(timestamp_to_annotations.keys())[:10],
        )  # Show first 10
    else:
        LOGGER.debug(
            'Frame %d found in annotations with %d entities',
            frame_count,
            len(gt_boxes_for_frame),
        )
    
    if len(gt_boxes_for_frame) > 0:
        LOGGER.debug('Ground truth entities:')
        for i, gt_box in enumerate(gt_boxes_pixel):
            entity_id = gt_box['entity_id']
            gt_label = gt_box['gt_vvad_label']
            bbox = gt_box['bbox_pixel']
            LOGGER.debug(
                '  GT %d: Entity %s, Label %s, Box %s',
                i,
                entity_id,
                gt_label,
                bbox,
            )
        # Draw all GT boxes on the frame for visual inspection of overlap with
        # the model's predicted face boxes.
        draw_gt_boxes_overlay(out_rgb, gt_boxes_pixel, width, height)
    else:
        LOGGER.debug('No ground truth entities for this frame!')
        # Show nearby frames that might have annotations
        nearby_frames = [f for f in timestamp_to_annotations.keys() if abs(f - frame_count) <= 5]
        if nearby_frames:
            LOGGER.debug('Nearby frames with annotations: %s', nearby_frames)
    
    if len(pred_boxes) > 0:
        LOGGER.debug('Predicted boxes:')
        for i, pred_box in enumerate(pred_boxes):
            pred_label = getattr(pred_box, 'class_name', 'Unknown')
            coords = pred_box.coordinates
            LOGGER.debug(
                '  Pred %d: Label %s, Box %s', i, pred_label, coords
            )
    
    # Timestamp-based denominator: every GT annotation on this evaluation frame
    # counts once, regardless of whether it receives a prediction.
    stats['total_eval_gt_annotations'] += len(gt_boxes_pixel)

    matches = match_boxes_to_gt(pred_boxes, gt_boxes_pixel, iou_threshold=iou_threshold)
    frame_correct = 0
    stats['total_matched'] += len(matches)
    
    LOGGER.debug('Matches found: %d', len(matches))
    
    # Show IoU for all entities that didn't get matched
    if len(gt_boxes_pixel) > 0 and len(pred_boxes) > 0:
        LOGGER.debug('IoU analysis for all entities:')
        for gt_idx, gt_box in enumerate(gt_boxes_pixel):
            entity_id = gt_box['entity_id']
            gt_label = gt_box['gt_vvad_label']
            best_iou = 0
            best_pred_idx = None
            
            for pred_idx, pred_box in enumerate(pred_boxes):
                iou = compute_iou(pred_box.coordinates, gt_box['bbox_pixel'])
                pred_label = getattr(pred_box, 'class_name', 'Unknown')
                LOGGER.debug(
                    '  Entity %s (GT %d, %s) vs Pred %d (%s): IoU = %.2f%%',
                    entity_id,
                    gt_idx,
                    gt_label,
                    pred_idx,
                    pred_label,
                    iou * 100.0,
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_iou < iou_threshold:
                LOGGER.debug(
                    '    → Best IoU for entity %s: %.2f%% (below threshold %.2f%%)',
                    entity_id,
                    best_iou * 100.0,
                    iou_threshold * 100.0,
                )
            else:
                LOGGER.debug(
                    '    → Best IoU for entity %s: %.2f%% (matched with Pred %d)',
                    entity_id,
                    best_iou * 100.0,
                    best_pred_idx,
                )
    
    # Track which GT indices have a correct VVAD label prediction; this is used
    # for the timestamp-based accuracy numerator (per-GT correctness).
    correct_gt_indices = set()

    for pred_idx, gt_idx, iou in matches:
        pred_box = pred_boxes[pred_idx]
        gt_box = gt_boxes_pixel[gt_idx]
        entity_id = gt_box['entity_id']
        gt_label = gt_box['gt_vvad_label']
        pred_label = getattr(pred_box, 'class_name', 'Unknown')
        
        LOGGER.debug(
            'Processing match: Pred %d (%s) -> GT %d (entity %s, %s) IoU: %.2f%%',
            pred_idx,
            pred_label,
            gt_idx,
            entity_id,
            gt_label,
            iou * 100.0,
        )
        
        if process_single_match(pred_box, gt_box, iou, width, height, stats, out_rgb):
            frame_correct += 1
            correct_gt_indices.add(gt_idx)
            LOGGER.debug('  Correct prediction')
        else:
            LOGGER.debug('  Wrong prediction')

    # Timestamp-based numerator: count GT annotations on this frame that have
    # at least one correct VVAD prediction. This is the per-frame contribution
    # to the final correct prediction count under the timestamp-based metric.
    stats['final_correct_predictions'] += len(correct_gt_indices)
    
    draw_frame_summary(out_rgb, frame_count, frame_correct, len(matches), height)
    LOGGER.debug(
        'Frame %d summary: %d/%d correct (%d matches total)',
        frame_count,
        frame_correct,
        len(matches),
        len(matches),
    )
    return frame_correct


def process_video_frames(cap, writer, pipeline, timestamp_to_annotations, eval_frame_indices,
                        width, height, dataset: AvaDataset, iou_threshold, stats):
    """Process all relevant video frames and perform VVAD evaluation.

    # Arguments
        cap: cv2.VideoCapture - Capture reading from the input video.
        writer: cv2.VideoWriter - Writer for annotated output frames.
        pipeline: Callable - VVAD detection pipeline callable.
        timestamp_to_annotations: Dictionary - Mapping from frame indices to
            annotations.
        eval_frame_indices: List of integers - Frame indices at which
            evaluation is performed.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        dataset: AvaDataset - Dataset instance used for label and box
            conversions.
        iou_threshold: Float - IoU threshold for matching predictions to GT
            boxes.
        stats: Dictionary - Global statistics dictionary being updated in
            place.

    Returns:
        int: Total number of frames processed.
    """
    # Start timing
    start_time = time.time()
    
    frame_count = 0
    max_frame = max(eval_frame_indices) if eval_frame_indices else 0
    
    while frame_count <= max_frame:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        
        frame_count = _process_single_frame(frame_bgr, frame_count, writer, pipeline,
                                           timestamp_to_annotations, eval_frame_indices,
                                           width, height, dataset, iou_threshold, stats)
    
    # End timing and store total prediction time
    end_time = time.time()
    stats['total_prediction_time'] = end_time - start_time
    
    return frame_count


def _process_single_frame(frame_bgr, frame_count, writer, pipeline, timestamp_to_annotations,
                         eval_frame_indices, width, height, dataset, iou_threshold, stats):
    """Process a single video frame and optionally evaluate it.

    # Arguments
        frame_bgr: numpy.ndarray - Input frame in BGR format as returned by
            OpenCV.
        frame_count: Integer - Index of the current frame.
        writer: cv2.VideoWriter - Writer used to write the output frame.
        pipeline: Callable - VVAD detection pipeline callable.
        timestamp_to_annotations: Dictionary - Mapping from frame indices to
            annotations.
        eval_frame_indices: List of integers - Frame indices at which
            evaluation is performed.
        width: Integer - Frame width in pixels.
        height: Integer - Frame height in pixels.
        dataset: AvaDataset - Dataset instance used for label and box
            conversions.
        iou_threshold: Float - IoU threshold for matching predictions to GT
            boxes.
        stats: Dictionary - Global statistics dictionary being updated in
            place.

    Returns:
        int: Index of the next frame to be processed.
    """
    try:
        if frame_bgr.size == 0:
            return frame_count + 1
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_rgb, pred_boxes = process_frame_predictions(frame_rgb, pipeline)

        # Track how many boxes the VVAD pipeline outputs before any matching.
        stats['total_pipeline_boxes'] += len(pred_boxes)
        if len(pred_boxes) > 0:
            stats['total_frame_predictions'] += 1
        
        if frame_count in eval_frame_indices:
            evaluate_frame(frame_count, pred_boxes, timestamp_to_annotations,
                         width, height, dataset, iou_threshold, stats, out_rgb)
        
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        writer.write(out_bgr)
        return frame_count + 1
        
    except Exception as e:
        LOGGER.error('Error processing frame %d: %s', frame_count, e)
        return frame_count + 1


def print_final_statistics(video_path: str, output_path: str, frame_count, fps,
                          eval_frame_indices, stats):
    """Print the final evaluation statistics after processing a video.

    # Arguments
        video_path: String - Path to the input video file.
        output_path: String - Path to the annotated output video file.
        frame_count: Integer - Total number of frames processed.
        fps: Float - Frames per second of the video.
        eval_frame_indices: List of integers - Frame indices that were
            evaluated.
        stats: Dictionary - Global statistics dictionary produced by the
            evaluation.

    Returns:
        None
    """
    # --- Overall/basic statistics ---
    total_matched = stats.get('total_matched', 0)
    total_correct = stats.get('total_correct', 0)
    speaking_count = stats.get('speaking_count', 0)
    not_speaking_count = stats.get('not_speaking_count', 0)
    total_pipeline_boxes = stats.get('total_pipeline_boxes', 0)
    total_frame_predictions = stats.get('total_frame_predictions', 0)
    total_eval_gt_annotations = stats.get('total_eval_gt_annotations', 0)
    final_correct_predictions = stats.get('final_correct_predictions', 0)
    total_time = stats.get('total_prediction_time', 0.0)

    # Detection-style accuracy: conditional on having a spatial match
    accuracy = (total_correct / total_matched) if total_matched > 0 else None
    # Timestamp-based accuracy: denominator is all GT annotations on evaluation
    # frames, numerator is those annotations with at least one correct VVAD
    # label prediction (final_correct_predictions).
    timestamp_accuracy = (
        final_correct_predictions / total_eval_gt_annotations
        if total_eval_gt_annotations > 0
        else None
    )

    LOGGER.info('')  # blank line
    LOGGER.info('Processing complete:')
    LOGGER.info('  Input:   %s', video_path)
    LOGGER.info('  Output:  %s', output_path)
    LOGGER.info(
        '  Total frames processed: %d @ %.2f FPS', frame_count, fps
    )
    LOGGER.info('  Evaluation frames: %d', len(eval_frame_indices))
    LOGGER.info('  Total frame predictions (frames with any pipeline boxes): %d',
                total_frame_predictions)
    LOGGER.info('  Total pipeline boxes (all frames): %d', total_pipeline_boxes)
    LOGGER.info('  Total GT pair matches (matched boxes): %d', total_matched)
    LOGGER.info(
        '  speaking: %d, not-speaking: %d',
        speaking_count,
        not_speaking_count,
    )

    if accuracy is None:
        LOGGER.info('  Detection-style Accuracy (label accuracy on spatially matched boxes): N/A (no matched boxes)')
    else:
        LOGGER.info(
            '  Detection-style Accuracy (label accuracy on spatially matched boxes): %.4f (%d/%d)',
            accuracy,
            total_correct,
            total_matched,
        )

    if timestamp_accuracy is None:
        LOGGER.info('  Timestamp-based Accuracy: N/A (no GT annotations on eval frames)')
    else:
        LOGGER.info(
            '  Timestamp-based Accuracy (per GT on eval frames): %.4f (%d/%d)',
            timestamp_accuracy,
            final_correct_predictions,
            total_eval_gt_annotations,
        )

    # Display total prediction time
    if total_time > 0:
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        if hours > 0:
            LOGGER.info(
                '  Total prediction time: %dh %dm %.2fs (%.2f seconds)',
                hours,
                minutes,
                seconds,
                total_time,
            )
        elif minutes > 0:
            LOGGER.info(
                '  Total prediction time: %dm %.2fs (%.2f seconds)',
                minutes,
                seconds,
                total_time,
            )
        else:
            LOGGER.info(
                '  Total prediction time: %.2f seconds', total_time
            )

    # --- Entity-level statistics ---
    entity_tracking = stats.get('entity_tracking', {})
    entity_predictions = entity_tracking.get('entity_predictions', {})
    entity_correct = entity_tracking.get('entity_correct', {})
    entity_matched = entity_tracking.get('entity_matched', {})
    entity_labels = entity_tracking.get('entity_labels', {})
    entity_gt_labels = entity_tracking.get('entity_gt_labels', {})
    entity_gt_total = entity_tracking.get('entity_gt_total', {})
    entity_gt_labels_all = entity_tracking.get('entity_gt_labels_all', {})

    LOGGER.debug('')  # blank line
    LOGGER.debug('Entity-level Accuracy (per-entity):')
    LOGGER.debug('=' * 60)
    
    # Show all entities (both matched and unmatched) in log file only
    all_entity_ids = sorted(set(list(entity_predictions.keys()) + list(entity_gt_total.keys())))
    
    for entity_id in all_entity_ids:
        if entity_id in entity_predictions:
            total = entity_predictions.get(entity_id, 0)
            correct_entity = entity_correct.get(entity_id, 0)
            matched_entity = entity_matched.get(entity_id, 0)
            acc_entity = correct_entity / matched_entity if matched_entity > 0 else 0
    
            pred_labels = entity_labels.get(entity_id, {})
            gt_labels_entity = entity_gt_labels.get(entity_id, {})
            gt_labels_all_entity = entity_gt_labels_all.get(entity_id, {})
            gt_total_entity = entity_gt_total.get(entity_id, 0)
    
            LOGGER.debug('  Entity %s:', entity_id)
            LOGGER.debug(
                '    Accuracy: %.4f (%d/%d)',
                acc_entity,
                correct_entity,
                matched_entity,
            )
            LOGGER.debug(
                '    Matched: %d/%d GT annotations',
                matched_entity,
                gt_total_entity,
            )
            LOGGER.debug(
                '    Predictions: speaking=%d, not-speaking=%d',
                pred_labels.get('speaking', 0),
                pred_labels.get('not-speaking', 0),
            )
            LOGGER.debug(
                '    Ground Truth (matched): speaking=%d, not-speaking=%d',
                gt_labels_entity.get('speaking', 0),
                gt_labels_entity.get('not-speaking', 0),
            )
            LOGGER.debug(
                '    Ground Truth (all from CSV): speaking=%d, not-speaking=%d',
                gt_labels_all_entity.get('speaking', 0),
                gt_labels_all_entity.get('not-speaking', 0),
            )
        else:
            # Entity exists in GT but has no matches
            gt_total_entity = entity_gt_total.get(entity_id, 0)
            gt_labels_all_entity = entity_gt_labels_all.get(entity_id, {})
            LOGGER.debug('  Entity %s:', entity_id)
            LOGGER.debug(
                '    No matches: 0/%d GT annotations matched', gt_total_entity
            )
            LOGGER.debug(
                '    Ground Truth (all from CSV): speaking=%d, not-speaking=%d',
                gt_labels_all_entity.get('speaking', 0),
                gt_labels_all_entity.get('not-speaking', 0),
            )

    # --- Entity summary ---
    all_entity_ids_set = set(entity_gt_total.keys())
    matched_entity_ids = set(entity_predictions.keys())

    total_entities_gt = len(all_entity_ids_set)
    total_entities_matched = len(matched_entity_ids)

    total_correct_entities = sum(entity_correct.values())
    total_matched_entities = sum(entity_matched.values())
    total_gt_annotations = sum(entity_gt_total.values())
    # Macro-averaged entity accuracy: mean of per-entity accuracies (each entity
    # weighted equally), not sum(correct)/sum(matched) which equals total accuracy.
    per_entity_accuracies = [
        entity_correct[eid] / entity_matched[eid]
        for eid in matched_entity_ids
        if entity_matched.get(eid, 0) > 0
    ]
    overall_entity_accuracy = (
        sum(per_entity_accuracies) / len(per_entity_accuracies)
        if per_entity_accuracies else 0
    )

    LOGGER.info('')  # blank line
    LOGGER.info('Entity Summary:')
    LOGGER.info('  Total unique entities in GT: %d', total_entities_gt)
    LOGGER.info('  Entities with matches: %d', total_entities_matched)
    LOGGER.info(
        '  Entities without matches: %d',
        total_entities_gt - total_entities_matched,
    )
    LOGGER.info('  Total GT annotations: %d', total_gt_annotations)
    LOGGER.info('  Total matched annotations: %d', total_matched_entities)
    LOGGER.info(
        '  Overall entity accuracy (macro over entities): %.4f (mean of %d per-entity accuracies)',
        overall_entity_accuracy,
        len(per_entity_accuracies),
    )

    unmatched_entities = all_entity_ids_set - matched_entity_ids
    if unmatched_entities:
        LOGGER.debug('')
        LOGGER.debug('  Unmatched entities (%d):', len(unmatched_entities))
        for entity_id in sorted(unmatched_entities)[:10]:  # Show first 10
            gt_count = entity_gt_total.get(entity_id, 0)
            LOGGER.debug(
                '    %s: %d GT annotations (no matches)', entity_id, gt_count
            )
        if len(unmatched_entities) > 10:
            LOGGER.debug(
                '    ... and %d more', len(unmatched_entities) - 10
            )


def evaluate_ava_video(video_path: str, output_path: str, dataset: AvaDataset, iou_threshold: float):
    """Evaluate the DetectVVAD model on a single AVA video.

    # Arguments
        video_path: String - Path to the input AVA video file.
        output_path: String - Path where the annotated output video will be
            written.
        dataset: AvaDataset - Dataset providing access to video metadata and
            annotations.
        iou_threshold: Float - IoU threshold for matching predicted boxes to
            GT boxes.

    Returns:
        None
    """
    pipeline, video_name, annots = setup_evaluation(video_path, dataset)
    cap, writer, fps, width, height = setup_video_io(video_path, output_path, dataset)
    
    timestamp_to_annotations, eval_frame_indices = prepare_evaluation_data(annots, fps, dataset)
    stats = initialize_statistics()
    
    # Initialize ground truth entity counts from all annotations
    entity_gt_total, entity_gt_labels_all = initialize_ground_truth_entity_counts(annots, dataset)
    stats['entity_tracking']['entity_gt_total'] = entity_gt_total
    stats['entity_tracking']['entity_gt_labels_all'] = entity_gt_labels_all
    
    try:
        frame_count = process_video_frames(
            cap,
            writer,
            pipeline,
            timestamp_to_annotations,
            eval_frame_indices,
            width,
            height,
            dataset,
            iou_threshold,
            stats,
        )
    finally:
        cap.release()
        try:
            writer.release()
        except Exception as e:
            LOGGER.warning('Error releasing video writer: %s', e)
    
    print_final_statistics(video_path, output_path, frame_count, fps, eval_frame_indices, stats)


def main():
    """Entry point for running VVAD evaluation on an AVA video from the CLI.

    Parses command-line arguments, prepares the dataset, selects the input and
    output paths, and invokes ``evaluate_ava_video``.

    Returns:
        None
    """
    args = parse_args()
    # Setup logging
    logger = setup_logging(args.log_file, args.enable_logging)
    if args.enable_logging:
        logger.info('Starting VVAD evaluation on AVA video')
        logger.info(f'Arguments: {vars(args)}')
    
    dataset = AvaDataset(root_dir=args.root_dir)
    
    if args.video is None:
        video_path = pick_first_annotated_video(dataset)
        if video_path is None:
            raise RuntimeError('No annotated AVA videos found')
        if args.enable_logging:
            logger.info(f'Using first annotated video: {video_path}')
    else:
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(dataset.video_dir, video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f'Video not found: {video_path}')
        if args.enable_logging:
            logger.info(f'Using specified video: {video_path}')

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(os.path.basename(video_path))
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_vvad.{args.output_ext}")
    
    if args.enable_logging:
        logger.info(f'Output will be saved to: {output_path}')

    evaluate_ava_video(
        video_path=video_path,
        output_path=output_path,
        dataset=dataset,
        iou_threshold=args.iou_threshold
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.info('Interrupted by user')
    except Exception as e:
        LOGGER.exception('Error during VVAD evaluation: %s', e)
        exit(1)
