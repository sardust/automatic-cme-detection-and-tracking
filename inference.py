import os
import math
import argparse

import cv2
import maxflow
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA

from vision_transformer import VisionTransformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="CME inference, segmentation, and geometric analysis"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--img_size", type=int, default=512, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")

    parser.add_argument(
        "--pca_thresh",
        type=int,
        default=6,
        help="Threshold for selecting high-confidence PCA region",
    )
    parser.add_argument(
        "--gray_img_thresh",
        type=int,
        default=100,
        help="Remove low-intensity pixels from the mask",
    )
    parser.add_argument(
        "--brightness_base",
        type=float,
        default=10.0,
        help="Brightness margin for removing weak false-positive regions",
    )
    parser.add_argument(
        "--close_to_base",
        type=float,
        default=2.0,
        help="Brightness tolerance for merging nearby pixels",
    )
    parser.add_argument(
        "--expand_tolerance",
        type=float,
        default=10.0,
        help="Brightness tolerance used in boundary expansion",
    )
    parser.add_argument(
        "--merge_kernel_size",
        type=int,
        default=64,
        help="Kernel size used to search the neighborhood of the main region",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=100,
        help="Minimum connected-component area kept for angle analysis",
    )
    return parser.parse_args()


def load_model(model_path, device, img_size, patch_size):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    load_msg = model.load_state_dict(state_dict, strict=False)

    if load_msg.missing_keys:
        print("Warning: missing keys when loading checkpoint:")
        print(load_msg.missing_keys)
    if load_msg.unexpected_keys:
        print("Warning: unexpected keys when loading checkpoint:")
        print(load_msg.unexpected_keys)

    model.to(device).eval()
    return model


def load_image(image_path, img_size, device):
    img = Image.open(image_path).convert("RGB")
    print(f"Original image size: {img.size}")

    if img.size != (img_size, img_size):
        raise ValueError(
            f"Input image size {img.size} does not match required size ({img_size}, {img_size})."
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    return img, input_tensor


def extract_patch_tokens(model, input_tensor):
    with torch.no_grad():
        patch_tokens = model.forward_tokens(input_tensor)
    return patch_tokens.squeeze(0).detach().cpu().numpy()


def build_pca_heatmap(patch_tokens, img_size, patch_size):
    h = w = img_size // patch_size
    expected_tokens = h * w

    n_tokens, dim = patch_tokens.shape
    if n_tokens != expected_tokens:
        raise ValueError(
            f"Unexpected token count: got {n_tokens}, expected {expected_tokens}. "
            "Check image size, patch size, and model definition."
        )

    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(patch_tokens)
    heatmap = principal_component.reshape(h, w)

    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_resized = cv2.resize(
        heatmap_uint8, (img_size, img_size), interpolation=cv2.INTER_NEAREST
    )
    return heatmap_resized


def graphcut_segmentation(gray_img, heatmap_resized, pca_thresh):
    heatmap_prob = heatmap_resized.astype(np.float32) / 255.0

    keep_mask = heatmap_resized <= pca_thresh
    heatmap_prob[~keep_mask] = 1.0

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes(gray_img.shape)

    # Lower probability indicates a higher likelihood of foreground (CME).
    g.add_grid_tedges(nodes, 1.0 - heatmap_prob, heatmap_prob)

    structure = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.int32)

    grad_weight = 50.0 / (np.abs(cv2.Laplacian(gray_img, cv2.CV_32F)) + 1.0)
    g.add_grid_edges(nodes, weights=grad_weight, structure=structure, symmetric=True)

    flow = g.maxflow()
    segmentation = g.get_grid_segments(nodes)
    refined_mask = np.logical_not(segmentation).astype(np.uint8) * 255
    return refined_mask, flow


def postprocess_mask(
    refined_mask,
    gray_img,
    gray_img_thresh,
    brightness_base,
    close_to_base,
    expand_tolerance,
    merge_kernel_size,
):
    refined_mask = refined_mask.copy()

    # Post-processing step 1: remove low-intensity regions.
    low_brightness_mask = gray_img < gray_img_thresh
    refined_mask[low_brightness_mask] = 0

    # Post-processing step 2: expand the mask to neighboring pixels with similar intensity.
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(refined_mask, kernel, iterations=1)
    border_candidates = np.logical_and(dilated_mask == 255, refined_mask == 0)

    foreground_pixels = gray_img[refined_mask == 255]
    if foreground_pixels.size > 0:
        mean_intensity = float(foreground_pixels.mean())
        similar_pixels = np.abs(gray_img.astype(np.float32) - mean_intensity) < expand_tolerance
        expand_region = np.logical_and(border_candidates, similar_pixels)
        refined_mask[expand_region] = 255

    # Post-processing step 3: keep the main region and absorb nearby similar regions.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)

    if num_labels > 1:
        main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        main_mask = (labels == main_label).astype(np.uint8) * 255

        main_brightness = gray_img[main_mask == 255]
        if main_brightness.size > 0:
            main_mean = float(np.mean(main_brightness))

            for label in range(1, num_labels):
                if label == main_label:
                    continue
                region_mask = labels == label
                region_brightness = gray_img[region_mask]
                if region_brightness.size == 0:
                    continue
                region_mean = float(np.mean(region_brightness))
                if region_mean < main_mean - brightness_base:
                    refined_mask[region_mask] = 0

            dilated_main = cv2.dilate(
                main_mask,
                np.ones((merge_kernel_size, merge_kernel_size), np.uint8),
                iterations=1,
            )

            neighbor_region = np.logical_and(refined_mask == 0, dilated_main == 255)
            close_mask = np.abs(gray_img.astype(np.float32) - main_mean) < close_to_base
            merge_mask = np.logical_and(neighbor_region, close_mask)
            refined_mask[merge_mask] = 255

    return refined_mask


def save_visualizations(output_dir, gray_img, refined_mask, heatmap_resized):
    heatmap_path = os.path.join(output_dir, "pca_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap_resized)
    print(f"Heatmap saved to {heatmap_path}")

    green_mask = np.zeros((refined_mask.shape[0], refined_mask.shape[1], 3), dtype=np.uint8)
    green_mask[:, :, 1] = refined_mask
    refined_path = os.path.join(output_dir, "refined_on_image.png")
    cv2.imwrite(refined_path, green_mask)
    print(f"Refined CME region (green on black) saved to {refined_path}")

    gray_output_path = os.path.join(output_dir, "gray_input.png")
    cv2.imwrite(gray_output_path, gray_img)
    print(f"Pure grayscale image saved to {gray_output_path}")

    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    contour_img = gray_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    contour_path = os.path.join(output_dir, "gray_with_contour.png")
    cv2.imwrite(contour_path, contour_img)
    print(f"Overlay with CME contour saved to {contour_path}")

    overlay_img = gray_rgb.copy()
    overlay_img[refined_mask == 255] = (0, 255, 0)
    overlay_path = os.path.join(output_dir, "gray_with_mask_overlay.png")
    cv2.imwrite(overlay_path, overlay_img)
    print(f"Grayscale image with CME green mask overlay saved to {overlay_path}")

    return {
        "heatmap_path": heatmap_path,
        "refined_path": refined_path,
        "gray_output_path": gray_output_path,
        "contour_path": contour_path,
        "overlay_path": overlay_path,
    }


def analyze_region(refined_mask, gray_img, min_area):
    area_pixels = int(np.count_nonzero(refined_mask))
    print(f"CME area: {area_pixels} pixels")

    height, width = gray_img.shape
    center = (width // 2, height // 2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask, connectivity=8)
    filtered_mask = np.zeros_like(refined_mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    region_coords = np.column_stack(np.where(filtered_mask > 0))
    if region_coords.size == 0:
        return {
            "area_pixels": area_pixels,
            "analysis_available": False,
            "message": "No CME region satisfying the minimum area threshold was detected, so angular and structural analysis cannot be performed.",
        }

    distances = []
    angles = []

    for y, x in region_coords:
        dx = x - center[0]
        dy = center[1] - y
        distances.append(math.hypot(dx, dy))
        angles.append((math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0)

    max_distance = float(max(distances))

    angles_array = np.sort(np.array(angles, dtype=np.float32))
    angle_diffs = np.diff(np.append(angles_array, angles_array[0] + 360.0))
    max_gap_index = int(np.argmax(angle_diffs))

    start_pa = float(angles_array[(max_gap_index + 1) % len(angles_array)] % 360.0)
    end_pa = float(angles_array[max_gap_index] % 360.0)
    angular_width = float((end_pa - start_pa) % 360.0)
    central_pa = float((start_pa + angular_width / 2.0) % 360.0)

    return {
        "area_pixels": area_pixels,
        "analysis_available": True,
        "max_distance": max_distance,
        "start_pa": start_pa,
        "end_pa": end_pa,
        "angular_width": angular_width,
        "central_pa": central_pa,
    }


def save_analysis(output_dir, analysis_result):
    output_file_path = os.path.join(output_dir, "CME_region_analysis.txt")

    if analysis_result["analysis_available"]:
        output_text = f"""CME Region Analysis Results:

Maximum Distance (Height): {analysis_result['max_distance']:.2f} pixels
Start PA: {analysis_result['start_pa']:.2f}°
End PA: {analysis_result['end_pa']:.2f}°
Angular Width: {analysis_result['angular_width']:.2f}°
Central PA: {analysis_result['central_pa']:.2f}°
"""
    else:
        output_text = analysis_result["message"] + "\n"

    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print("Analysis results saved to:", output_file_path)
    return output_file_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.model_path, device, args.img_size, args.patch_size)
    img, input_tensor = load_image(args.image_path, args.img_size, device)

    # Extract patch tokens directly without using a forward hook.
    patch_tokens = extract_patch_tokens(model, input_tensor)
    heatmap_resized = build_pca_heatmap(patch_tokens, args.img_size, args.patch_size)

    gray_img = np.array(img.convert("L"))
    refined_mask, flow = graphcut_segmentation(gray_img, heatmap_resized, args.pca_thresh)
    print(f"GraphCut flow (original image): {flow:.2f}")

    refined_mask = postprocess_mask(
        refined_mask=refined_mask,
        gray_img=gray_img,
        gray_img_thresh=args.gray_img_thresh,
        brightness_base=args.brightness_base,
        close_to_base=args.close_to_base,
        expand_tolerance=args.expand_tolerance,
        merge_kernel_size=args.merge_kernel_size,
    )

    save_visualizations(args.output_dir, gray_img, refined_mask, heatmap_resized)

    analysis_result = analyze_region(refined_mask, gray_img, args.min_area)
    save_analysis(args.output_dir, analysis_result)

    if analysis_result["analysis_available"]:
        print(f"Start PA: {analysis_result['start_pa']:.2f}°")
        print(f"End PA: {analysis_result['end_pa']:.2f}°")
        print(f"Angular Width: {analysis_result['angular_width']:.2f}°")
        print(f"Central PA: {analysis_result['central_pa']:.2f}°")


if __name__ == "__main__":
    main()