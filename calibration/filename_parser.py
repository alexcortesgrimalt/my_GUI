# filename_parser.py

from collections import defaultdict
import os


def parse_filename(filename):
    """
    Parse filename into physical parameters.

    Expected format:
    2keV_120um_HC_5000x.tiff
    """

    name = filename.lower().replace(".tiff", "").replace(".tif", "")
    parts = name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Filename format not recognized: {filename}")

    return {
        "energy": parts[0],        # e.g. 2keV
        "aperture": parts[1],      # e.g. 120um
        "condition": parts[2],     # e.g. HC
        "magnification": parts[3]  # e.g. 5000x
    }


def extract_condition(filename):
    """
    Return grouping key based on:
    (energy, aperture, condition)
    """
    p = parse_filename(filename)
    return f"{p['energy']}_{p['aperture']}_{p['condition']}"


def group_results(results):
    """
    Group results by experimental condition.
    """
    grouped = defaultdict(list)

    for r in results:
        key = extract_condition(r['filename'])
        grouped[key].append(r)

    return grouped


def aggregate_grouped_results(grouped):
    """
    Compute mean and std across grouped measurements.
    """
    import numpy as np

    summary = []

    for key, entries in grouped.items():
        means = [e.get('mean_in') for e in entries if e.get('mean_in') is not None]
        stds = [e.get('std_in') for e in entries if e.get('std_in') is not None]

        if len(means) == 0:
            continue

        mean_global = np.mean(means)
        std_global = np.std(means)

        summary.append({
            "condition": key,
            "mean_current_nA": mean_global,
            "std_current_nA": std_global,
            "n_measurements": len(means)
        })

    return summary

def get_magnification(file_path):
    filename = os.path.basename(file_path)  # 👈 CRITICAL FIX
    p = parse_filename(filename)

    mag_string = p["magnification"].lower()
    return int(mag_string.replace("x", ""))