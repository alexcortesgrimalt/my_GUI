import os
from calibration.faraday_calibrator import FaradayCalibrator


def main():
    # 👉 CHANGE THIS PATH
    folder = r"C:\Users\piazza\Desktop\Valerio\Scientific\Research\Management_Lab\Setups\IMINA Platform @CIME\CalibrationCurrent_2026.03\2026.03.11_CalibrationFaradayCup\DISS test\Calib images\testanalysis"

    calibrator = FaradayCalibrator()

    print("\n--- Processing files ---\n")
    calibrator.process_folder(folder)

    print("\n--- Raw results ---\n")
    for r in calibrator.results:
        print(r)

    print("\n--- Grouped summary ---\n")
    summary = calibrator.summarize()

    for s in summary:
        print(
            f"{s['condition']} | "
            f"Mean: {s['mean_current_nA']:.3e} nA | "
            f"Std: {s['std_current_nA']:.3e} | "
            f"N: {s['n_measurements']}"
        )


if __name__ == "__main__":
    main()