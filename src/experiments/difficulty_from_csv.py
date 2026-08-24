import csv
from collections import defaultdict
from statistics import mean,variance
from paz.backend.boxes import compute_iou,compute_ious
import numpy as np


def load_ava_as(csv_path):
    """Load Active Speaker CSV annotations.

    # Arguments
        csv_path: String. Path to a single AVA ActiveSpeaker CSV file.

    # Returns
        Dictionary mapping video_id to a list of annotation dictionaries.
        Each annotation dictionary contains:
            - 'ts': Float. Timestamp.
            - 'box': Tuple of four floats (x1, y1, x2, y2).
            - 'label': String. Active speaker label.
            - 'ent': String. Entity / track identifier.
    """
    by_vid = defaultdict(list)
    with open(csv_path, newline='') as f:
        r = csv.reader(f)
        for row in r:
            vid, ts = row[0], float(row[1])
            x1, y1, x2, y2 = map(float, row[2:6])
            label = row[6].strip()
            ent = row[7].strip()
            by_vid[vid].append({
                'ts': ts, 'box': (x1, y1, x2, y2),
                'label': label, 'ent': ent
            })
    for vid in by_vid:
        by_vid[vid].sort(key=lambda d: (d['ts'], d['ent']))
    return by_vid

def compute_difficulty(rows):
    """Compute difficulty metrics for a single video.

    # Arguments
        rows: List of dictionaries for one video. Each dictionary contains:
            - 'ts': Float. Timestamp.
            - 'box': Tuple of four floats (x1, y1, x2, y2).
            - 'label': String. Active speaker label.
            - 'ent': String. Entity / track identifier.

    # Returns
        Dictionary with:
            diversity: Int. Number of unique entity IDs(proxy for people).
            interactivity_avg: Float. Average number of faces per frame.
            interactivity_max: Int. Maximum number of faces in any frame.
            audibility_sna_share: Float. Ratio of NOT_AUDIBLE speech among all
                speaking labels in [0, 1].
            occlusion: Float. Average pairwise IoU across all face pairs.
            dynamics_mean: Float. Mean motion measure (1 - IoU) over time.
            dynamics_var: Float. Variance of the motion measure.
    """
    # group by timestamp
    by_ts = defaultdict(list)
    entities = set()
    for r in rows:
        by_ts[r['ts']].append(r)
        entities.add(r['ent'])

    # interactivity
    faces_per_frame = [len(v) for v in by_ts.values()]
    interactivity_avg = mean(faces_per_frame) if faces_per_frame else 0.0
    interactivity_max = max(faces_per_frame) if faces_per_frame else 0

    # diversity
    diversity = len(entities)

    # audibility: percentage of SPEAKING_NOT_AUDIBLE among all speaking
    speak_na = [
        r for r in rows
        if r['label'].startswith('SPEAKING') and 'NOT_AUDIBLE' in r['label']
    ]
    speak_audible = [
        r for r in rows
        if r['label'].startswith('SPEAKING') and 'AUDIBLE' in r['label']
    ]
    total_speaking = len(speak_na) + len(speak_audible)
    audibility_sna_share = (
        len(speak_na) / total_speaking if total_speaking else 0.0
    )


    # dynamics: mean (1 - IoU) per entity over successive frames
    by_ent = defaultdict(list)
    for r in rows:
        by_ent[r['ent']].append(r)
    dynamics_terms = []
    for ent, seq in by_ent.items():
        seq.sort(key=lambda d: d['ts'])
        for i in range(1, len(seq)):
            # inside the loop
            prev_box = np.array(seq[i-1]['box'], dtype=float)
            curr_box = np.array([seq[i]['box']], dtype=float)  # shape (1, 4)
            iou_val = float(compute_iou(prev_box, curr_box)[0])
            dynamics_terms.append(1.0 - iou_val)

    if dynamics_terms:
        dynamics_mean = mean(dynamics_terms)
        dynamics_var = variance(dynamics_terms) if len(dynamics_terms) > 1 else 0.0
    else:
        dynamics_mean = 0.0
        dynamics_var = 0.0

    # occlusion: average pairwise IoU per frame when >=2 faces
    occlusion_terms = []

    for ts, dets in by_ts.items():
        n = len(dets)
        if n < 2:
            continue

        boxes = np.array([d['box'] for d in dets], dtype=float)  # (n, 4)
        ious = compute_ious(boxes, boxes)  # (n, n)

        # take only upper triangle (i<j) to avoid self and double counting
        upper = ious[np.triu_indices(n, k=1)]
        if upper.size > 0:
            occlusion_terms.append(float(upper.mean()))

    occlusion = mean(occlusion_terms) if occlusion_terms else 0.0

    return dict(
        diversity=diversity,
        interactivity_avg=interactivity_avg,
        interactivity_max=interactivity_max,
        audibility_sna_share=audibility_sna_share,
        occlusion=occlusion,
        dynamics_mean=dynamics_mean,
        dynamics_var=dynamics_var
    )

if __name__ == "__main__":
    import os
    import json
    import glob
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute difficulty metrics from AVA ActiveSpeaker CSV files.')
    parser.add_argument(
        'path',
        help='Path to a single AVA CSV file or a directory containing multiple CSV files.')
    parser.add_argument(
        '--output', '-o',
        help=('Output JSON file path. '
              'Defaults to "difficulty_ava_test.json" next to this script.'))
    args = parser.parse_args()

    path = args.path
    out = {}

    def process_csv(csv_path):
        by_vid = load_ava_as(csv_path)
        for vid, rows in by_vid.items():
            out[vid] = compute_difficulty(rows)

    if os.path.isdir(path):
        # Process every *.csv in the folder
        for csv_path in sorted(glob.glob(os.path.join(path, "*.csv"))):
            process_csv(csv_path)
    else:
        # Single CSV file
        process_csv(path)

    # Min–max normalization across all videos per metric
    if out:
        metric_names = [
        'diversity',
        'interactivity_avg',
        'interactivity_max',
        'audibility_sna_share',
        'occlusion',
        'dynamics_mean',
        'dynamics_var'
        ]

        # collect min / max per metric
        mins = {}
        maxs = {}
        for m in metric_names:
            vals = [v[m] for v in out.values() if m in v]
            if not vals:
                continue
            mins[m] = min(vals)
            maxs[m] = max(vals)

        def minmax(x, xmin, xmax):
            if xmax == xmin:
                return 0.0
            return (x - xmin) / (xmax - xmin)
        
        # add normalized versions with *_norm suffix
        for vid, metrics in out.items():
            for m in metric_names:
                if m in metrics and m in mins:
                    metrics[m + '_norm'] = minmax(metrics[m], mins[m], maxs[m])


    # Save JSON next to this script by default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = args.output or os.path.join(script_dir, "difficulty_ava_test.json")

    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved difficulty metrics to: {out_file}")

