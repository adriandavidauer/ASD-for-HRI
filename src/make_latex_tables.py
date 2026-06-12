"""Build a LaTeX comparison table (one row per model) from the scoring outputs.

For every model:
    Accuracy / F1 / Precision / Recall  come from the ``micro_average`` row that
        stats.py writes at the end of <STATS_BASE>/predictions_<suffix>/aggregate_results.csv
    FPS  is the mean of ``fps_processed`` over the videos in
        <PRED_BASE>/predictions_<suffix>/aggregate_time.csv

The folder suffix for each display label is the label lowercased with every
non-alphanumeric character removed (e.g. "CNN2Plus1D_Filters" -> "cnn2plus1dfilters").

Uses only the Python standard library (no pandas), so it runs in any env.
"""

import os
import re
import csv

STATS_BASE = "/Data/data/stats"   # holds predictions_<suffix>/aggregate_results.csv
PRED_BASE  = "/Data/data"         # holds predictions_<suffix>/aggregate_time.csv
OUTPUT_TEX = "/Data/data/stats/model_comparison.tex"

# Display label -> shown verbatim in the table. Order is preserved.
MODELS = [
    "CNN2Plus1D",
    "CNN2Plus1D_Filters",
    "CNN2Plus1D_Layers",
    "CNN2Plus1D_Light",
    "LipShape",
    "FaceShape",
]

# Column order requested for the table (label -> value formatter).
METRIC_COLS = ["Accuracy", "FPS", "F1", "Precision", "Recall"]
FORMATS = {
    "Accuracy": "{:.4f}",
    "FPS": "{:.2f}",
    "F1": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
}


def folder_suffix(label):
    """Lowercase the label and strip every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]", "", label.lower())


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_row(label):
    suffix = folder_suffix(label)
    results_csv = os.path.join(STATS_BASE, f"predictions_{suffix}", "aggregate_results.csv")
    time_csv = os.path.join(PRED_BASE, f"predictions_{suffix}", "aggregate_time.csv")

    # Micro-average row written by stats.py (take the last if appended repeatedly).
    micro = None
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("video_id") == "micro_average":
                micro = row
    if micro is None:
        raise ValueError(f"No 'micro_average' row found in {results_csv}")

    # Mean FPS across this model's videos.
    fps_vals = []
    with open(time_csv, newline="") as f:
        for row in csv.DictReader(f):
            v = _to_float(row.get("fps_processed"))
            if v is not None:
                fps_vals.append(v)
    fps = sum(fps_vals) / len(fps_vals) if fps_vals else float("nan")

    return {
        "Model": label,
        "Accuracy": float(micro["accuracy"]),
        "FPS": fps,
        "F1": float(micro["f1"]),
        "Precision": float(micro["precision"]),
        "Recall": float(micro["recall"]),
    }


def to_latex(rows):
    """Render the rows as a booktabs LaTeX table (needs \\usepackage{booktabs})."""
    col_fmt = "l" + "r" * len(METRIC_COLS)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model Performace across UniTalk Dataset (val split)}",
        r"\label{tab:model_comparison}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        "Model & " + " & ".join(METRIC_COLS) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        cells = [FORMATS[c].format(r[c]) for c in METRIC_COLS]
        lines.append(f"{r['Model']} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    rows = [model_row(m) for m in MODELS]
    latex = to_latex(rows)

    with open(OUTPUT_TEX, "w") as f:
        f.write(latex)
    print(f"LaTeX table written to {OUTPUT_TEX}\n")
    print(latex)


if __name__ == "__main__":
    main()
