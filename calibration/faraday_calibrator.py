import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2
import pandas as pd

from calibration.image_processing import detect_faraday_region, refine_with_ebic
from calibration.statistics import compute_stats
from calibration.filename_parser import group_results, aggregate_grouped_results
from image_handler import SEMDataManager
from calibration.filename_parser import (
    group_results,
    aggregate_grouped_results
)
from calibration.filename_parser import parse_filename
from calibration.filename_parser import get_magnification
from calibration.image_processing import circular_core_mask


class FaradayCalibrator:

    def __init__(self):
        self.results = []

    def process_file(self, file_path):

        manager = SEMDataManager()
        print(f"Processing: {file_path}")

        if not manager.load_file(file_path):
            return

        sem = manager.sem_data
        ebic = manager.current_map

        if sem is None or ebic is None:
            return

        m = manager.metadata.data if hasattr(manager.metadata, "data") else manager.metadata

        required = ["Contrast", "EffectiveAmpGain", "OutputOffset", "InputOffset"]
        missing = [k for k in required if k not in m]
        if missing:
            raise ValueError(f"[Calibration] Missing metadata {missing} in {file_path}")

        mag = get_magnification(file_path)

        ebic_clean = np.nan_to_num(ebic)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # ============================================================
        # MASKING
        # ============================================================
        mask_inner = None
        mask_outside = None

        if mag < 1000:

            img = ndi.gaussian_filter(ebic_clean, sigma=2)
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            circles = cv2.HoughCircles(
                img_norm,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=img.shape[0] // 4,
                param1=100,
                param2=30,
                minRadius=int(min(img.shape) * 0.2),
                maxRadius=int(min(img.shape) * 0.6)
            )

            if circles is None:
                raise RuntimeError(f"Faraday cup not detected in {file_path}")

            circles = np.round(circles[0]).astype(int)
            x, y, r = circles[0]

            Y, X = np.ogrid[:ebic.shape[0], :ebic.shape[1]]

            mask_inner = (X - x)**2 + (Y - y)**2 <= (0.9 * r)**2
            mask_outer = (X - x)**2 + (Y - y)**2 <= (1.1 * r)**2
            mask_outside = mask_outer & (~mask_inner)

        else:
            mask_inner = np.ones_like(ebic_clean, dtype=bool)
            mask_outside = np.zeros_like(ebic_clean, dtype=bool)

        # ============================================================
        # STATISTICS
        # ============================================================

        values_in = ebic[mask_inner]
        values_out = ebic[mask_outside]

        mean_in = np.mean(values_in) if values_in.size > 0 else None
        std_in = np.std(values_in) if values_in.size > 0 else None

        mean_out = np.mean(values_out) if values_out.size > 0 else None
        std_out = np.std(values_out) if values_out.size > 0 else None

        parsed = parse_filename(base_name)

        self.results.append({
            "filename": base_name,
            "energy": parsed["energy"],
            "aperture": parsed["aperture"],
            "condition": parsed["condition"],
            "magnification": mag,
            "mean_in": mean_in,
            "std_in": std_in,
            "mean_out": mean_out,
            "std_out": std_out
        })

        # ============================================================
        # SAVE OVERLAY IMAGES (NO PLOTS)
        # ============================================================

        out_dir = os.path.join(os.path.dirname(file_path), "calibration_outputs")
        os.makedirs(out_dir, exist_ok=True)

        # --- INNER OVERLAY ---
        plt.figure(figsize=(6, 5))
        plt.imshow(ebic, cmap="inferno")
        plt.imshow(mask_inner, cmap="gray", alpha=0.3)
        plt.title(f"{base_name} | INNER (90%)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_inner.png"), dpi=200)
        plt.close()

        # --- OUTER OVERLAY ---
        plt.figure(figsize=(6, 5))
        plt.imshow(ebic, cmap="inferno")
        plt.imshow(mask_outside, cmap="gray", alpha=0.3)
        plt.title(f"{base_name} | OUTER (110%)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_outer.png"), dpi=200)
        plt.close()

    def export_csv(self, folder_path):


        if not hasattr(self, "results") or len(self.results) == 0:
            print("[Calibration] No results to export.")
            return

        df = pd.DataFrame(self.results)

        # ============================================================
        # GROUP KEY
        # ============================================================
        df["group"] = (
            df["energy"].astype(str)
            + "_" + df["aperture"].astype(str)
            + "_" + df["condition"].astype(str)
        )

        # ============================================================
        # RAW DATA (per image)
        # ============================================================
        raw_df = df.copy()
        raw_df["type"] = "raw"

        raw_df = raw_df.rename(columns={
            "filename": "Filename",
            "energy": "Beam Energy",
            "aperture": "Aperture",
            "condition": "Condition",
            "magnification": "Magnification",
            "mean_in": "Mean Current (Inner Cup)",
            "std_in": "Std Dev (Inner Cup)",
            "mean_out": "Mean Current (Outside Cup)",
            "std_out": "Std Dev (Outside Cup)",
            "group": "Group"
        })

        # ============================================================
        # SUMMARY DATA (grouped)
        # ============================================================
        summary = df.groupby("group").agg({
            "mean_in": ["mean", "std"],
            "mean_out": ["mean", "std"]
        })

        summary.columns = ["_".join(col) for col in summary.columns]
        summary = summary.reset_index()

        summary["type"] = "summary"

        summary = summary.rename(columns={
            "group": "Group",
            "mean_in_mean": "Mean Current (Inner Cup)",
            "mean_in_std": "Std Current (Inner Cup)",
            "mean_out_mean": "Mean Current (Outside Cup)",
            "mean_out_std": "Std Current (Outside Cup)"
        })

        # ============================================================
        # ALIGN COLUMNS
        # ============================================================
        for col in raw_df.columns:
            if col not in summary.columns:
                summary[col] = None

        for col in summary.columns:
            if col not in raw_df.columns:
                raw_df[col] = None

        # ============================================================
        # FINAL MERGE
        # ============================================================
        final_df = pd.concat([raw_df, summary], ignore_index=True, sort=False)

        # ============================================================
        # COLUMN ORDER (structured output)
        # ============================================================
        column_order = [
            "type",
            "Filename",
            "Beam Energy",
            "Aperture",
            "Condition",
            "Magnification",
            "Group",
            "Mean Current (Inner Cup)",
            "Std Dev (Inner Cup)",
            "Mean Current (Outside Cup)",
            "Std Dev (Outside Cup)"
        ]

        # keep only existing columns in order
        final_df = final_df[[c for c in column_order if c in final_df.columns]]

        # ============================================================
        # SAVE
        # ============================================================
        out_csv = os.path.join(folder_path, "calibration_results.csv")
        final_df.to_csv(out_csv, index=False)

        print(f"[Calibration] CSV saved: {out_csv}")

    def process_folder(self, folder_path):
        import os

        self.results = []

        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".tif", ".tiff"))
        ]

        files.sort()

        for file in files:
            full_path = os.path.join(folder_path, file)
            self.process_file(full_path)

        self.export_csv(folder_path)

        return self.results

    def summarize(self):
        grouped = group_results(self.results)
        return aggregate_grouped_results(grouped)

