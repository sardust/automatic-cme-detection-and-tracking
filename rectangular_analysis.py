import os
import argparse
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a segmented CME mask in rectangular coordinates and compute "
            "angular/radial geometric parameters."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the segmented overlay image (for example, gray_with_mask_overlay.png).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the rectangular plot and summary file.",
    )
    parser.add_argument(
        "--solar_radius_pixels",
        type=float,
        default=60.0,
        help="Solar radius in pixels used to convert radius to R / Rsun.",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=20,
        help="Minimum connected-component area to keep in the green mask.",
    )
    parser.add_argument(
        "--scatter_size",
        type=float,
        default=1.0,
        help="Marker size for the rectangular scatter plot.",
    )
    parser.add_argument(
        "--x_padding",
        type=float,
        default=10.0,
        help="Extra padding added to the x-axis limits when the CME spans across 0 degrees.",
    )
    return parser.parse_args()



def load_image(input_path: str) -> np.ndarray:
    image = Image.open(input_path).convert("RGB")
    return np.array(image)



def extract_green_mask(image_np: np.ndarray) -> np.ndarray:
    return (
        (image_np[:, :, 0] == 0)
        & (image_np[:, :, 1] == 255)
        & (image_np[:, :, 2] == 0)
    )



def filter_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    filtered_mask = np.zeros_like(mask_uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_size:
            filtered_mask[labels == label] = 1

    return filtered_mask.astype(bool)



def compute_geometry(
    cleaned_mask: np.ndarray,
    solar_radius_pixels: float,
) -> Dict[str, np.ndarray]:
    y_indices, x_indices = np.where(cleaned_mask)
    if y_indices.size == 0:
        raise ValueError(
            "No valid green-mask pixels were found after connected-component filtering."
        )

    height, width = cleaned_mask.shape
    center_x, center_y = width // 2, height // 2

    dx = x_indices - center_x
    dy = y_indices - center_y
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_over_rsun = r / solar_radius_pixels
    angles = (np.arctan2(dx, -dy) * 180.0 / np.pi) % 360.0

    return {
        "x_indices": x_indices,
        "y_indices": y_indices,
        "r_over_rsun": r_over_rsun,
        "angles": angles,
        "center_x": center_x,
        "center_y": center_y,
    }



def angles_for_plot(angles: np.ndarray) -> np.ndarray:
    if angles.size == 0:
        return angles

    if (angles.max() - angles.min()) > 180.0:
        return np.where(angles > 180.0, angles - 360.0, angles)
    return angles.copy()



def summarize_angles(angles: np.ndarray, r_over_rsun: np.ndarray) -> Dict[str, float]:
    angles_wrapped = (angles + 360.0) % 360.0
    min_angle = float(np.min(angles_wrapped))
    max_angle = float(np.max(angles_wrapped))

    if (max_angle - min_angle) > 180.0:
        adjusted_angles = np.where(angles_wrapped < 180.0, angles_wrapped + 360.0, angles_wrapped)
        min_angle = float(np.min(adjusted_angles))
        max_angle = float(np.max(adjusted_angles))
        cpa = (min_angle + (max_angle - min_angle) / 2.0) % 360.0
        tstart = min_angle % 360.0
        tend = max_angle % 360.0
        aw = (max_angle - min_angle) % 360.0
    else:
        cpa = (min_angle + (max_angle - min_angle) / 2.0) % 360.0
        tstart = min_angle
        tend = max_angle
        aw = (max_angle - min_angle) % 360.0

    rmax = float(np.max(r_over_rsun))
    CPA = (360.0 - cpa) % 360.0

    return {
        "Tstart": round(tstart, 2),
        "Tend": round(tend, 2),
        "AW": round(aw, 2),
        "CPA": round(CPA, 2),
        "Rmax": round(rmax, 2),
    }



def save_rectangular_plot(
    output_dir: str,
    angles_plot: np.ndarray,
    r_over_rsun: np.ndarray,
    scatter_size: float,
    x_padding: float,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "polar_to_rectangular.png")

    plt.figure(figsize=(12, 6))
    plt.scatter(angles_plot, r_over_rsun, s=scatter_size, color="green")

    if angles_plot.size > 0 and np.any(angles_plot < 0):
        xmin = float(np.min(angles_plot) - x_padding)
        xmax = float(np.max(angles_plot) + x_padding)
        plt.xlim(xmin, xmax)
        plt.axvline(0, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    else:
        plt.xlim(0, 360)

    plt.xlabel("Angle (degrees)")
    plt.ylabel("R / Rsun")
    plt.title("CME Green Pixels: Rectangular Coordinate Plot")
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path



def save_summary(output_dir: str, summary: Dict[str, float]) -> str:
    summary_path = os.path.join(output_dir, "cme_angle_summary.txt")
    summary_text = f"""CME Green Pixel Analysis Summary
--------------------------------
Tstart (Start Angle): {summary['Tstart']}°
Tend   (End Angle):   {summary['Tend']}°
AW     (Angular Width): {summary['AW']}°
CPA    (Central Position Angle): {summary['CPA']}°
Rmax   (Maximum Height): {summary['Rmax']} Rsun
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return summary_path



def main() -> None:
    args = parse_args()

    image_np = load_image(args.input_path)
    green_mask = extract_green_mask(image_np)
    cleaned_mask = filter_small_components(green_mask, args.min_component_size)

    geometry = compute_geometry(cleaned_mask, args.solar_radius_pixels)
    plot_angles = angles_for_plot(geometry["angles"])
    summary = summarize_angles(geometry["angles"], geometry["r_over_rsun"])

    plot_path = save_rectangular_plot(
        args.output_dir,
        plot_angles,
        geometry["r_over_rsun"],
        args.scatter_size,
        args.x_padding,
    )
    summary_path = save_summary(args.output_dir, summary)

    print(f"Rectangular plot saved to: {plot_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Tstart: {summary['Tstart']}°")
    print(f"Tend: {summary['Tend']}°")
    print(f"AW: {summary['AW']}°")
    print(f"CPA : {summary['CPA']}°")
    print(f"Rmax: {summary['Rmax']} Rsun")


if __name__ == "__main__":
    main()
