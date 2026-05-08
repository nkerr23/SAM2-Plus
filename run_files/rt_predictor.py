import os
import sys
import torch
import cv2
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from run_files.box_inf import build_predictor, track_boundingbox
from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from tools.visualization import show_mask, show_box, show_points
from run_files.img_input import motion_video

#---- INPUT ------------------------------------------------------------------------------

'''
input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152'
manual_boundbox = '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152/liver'
output_root = "./my_data_box" 
'''

#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------

#---- helper functions ------------------------------------------------------------------

def initial_coords(coords_dir: str):
    first_file = natsorted(Path(coords_dir).glob("*"))[0]
    box = np.load(first_file)
    print(box)
    return box

def load_manual_boxes(manual_dir: str) -> dict[int, np.ndarray]:
    files = natsorted(Path(manual_dir).glob("*.npy"))
    return {idx: np.load(f).astype(float) for idx, f in enumerate(files)}

def save_boxcoords(predicted: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for idx, box in predicted.items():
        np.save(os.path.join(output_dir, f"{idx:04d}.npy"), box)
    print(f"Saved {len(predicted)} boxes to {output_dir}")

def box_iou(box1_xywh, box2_xywh):
    x1, y1, w1, h1 = box1_xywh
    x2, y2, w2, h2 = box2_xywh

    # convert to xyxy
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = w1*h1 + w2*h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def save_vid(predicted: dict, input_dir: str, output_dir: str):
    npy_files = natsorted(Path(input_dir).glob("*.npy"))
    os.makedirs(output_dir, exist_ok=True)

    for out_frame_idx, npy_path in enumerate(tqdm(npy_files, desc="Saving visualization")):
        if out_frame_idx not in predicted:
            continue

        # load and normalize frame
        frame = normalize(np.load(npy_path))  # already BGR uint8

        # draw box directly with OpenCV — no matplotlib, no size drift
        box_xywh = predicted[out_frame_idx].flatten()
        box_xyxy = np_box_xywh_to_xyxy(np.array(box_xywh)).astype(int)
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        cv2.imwrite(
            os.path.join(output_dir, f"{out_frame_idx:05d}_withbox.png"),
            frame
        )

    print(f"Saved {len(predicted)} frames to {output_dir}")

def evaluate_and_plot(predicted: dict, manual: dict, output_dir: str):
    """
    Compare predicted boxes to manual annotations and plot metrics.
    
    Parameters
    ----------
    predicted : dict[int, np.ndarray]   frame_idx -> box xywh
    manual    : dict[int, np.ndarray]   frame_idx -> box xywh
    output_dir: str                     where to save plots
    """
    common = sorted(set(predicted.keys()) & set(manual.keys()))
    if not common:
        print("No overlapping frames between predicted and manual boxes.")
        return

    ious, mses, center_dists = [], [], []
    pred_cx, pred_cy, man_cx, man_cy = [], [], [], []
    pred_tl_x, pred_tl_y, pred_br_x, pred_br_y = [], [], [], []
    man_tl_x,  man_tl_y,  man_br_x,  man_br_y  = [], [], [], []

    for idx in common:
        p = predicted[idx].flatten()
        m = manual[idx].flatten()

        # IoU
        ious.append(box_iou(p, m))

        # MSE over all 4 coords
        mses.append(np.mean((p - m) ** 2))

        # centers
        pcx, pcy = p[0] + p[2]/2, p[1] + p[3]/2
        mcx, mcy = m[0] + m[2]/2, m[1] + m[3]/2
        pred_cx.append(pcx); pred_cy.append(pcy)
        man_cx.append(mcx);  man_cy.append(mcy)
        center_dists.append(np.sqrt((pcx-mcx)**2 + (pcy-mcy)**2))

        # corners (top-left, bottom-right)
        pred_tl_x.append(p[0]); pred_tl_y.append(p[1])
        pred_br_x.append(p[0]+p[2]); pred_br_y.append(p[1]+p[3])
        man_tl_x.append(m[0]);  man_tl_y.append(m[1])
        man_br_x.append(m[0]+m[2]); man_br_y.append(m[1]+m[3])

    frames = common
    os.makedirs(output_dir, exist_ok=True)

    # --- plot 1: IoU over time ---
    plt.figure(figsize=(10, 3))
    plt.plot(frames, ious)
    plt.axhline(np.mean(ious), color='r', linestyle='--', label=f'mean={np.mean(ious):.3f}')
    plt.xlabel("Frame"); plt.ylabel("IoU")
    plt.title("IoU over time")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou.png"), dpi=150)
    plt.close()

    # --- plot 2: MSE over time ---
    plt.figure(figsize=(10, 3))
    plt.plot(frames, mses)
    plt.axhline(np.mean(mses), color='r', linestyle='--', label=f'mean={np.mean(mses):.3f}')
    plt.xlabel("Frame"); plt.ylabel("MSE")
    plt.title("MSE over time")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse.png"), dpi=150)
    plt.close()

    # --- plot 3: center distance over time ---
    plt.figure(figsize=(10, 3))
    plt.plot(frames, center_dists)
    plt.axhline(np.mean(center_dists), color='r', linestyle='--', label=f'mean={np.mean(center_dists):.3f}')
    plt.xlabel("Frame"); plt.ylabel("Pixels")
    plt.title("Center distance over time")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "center_dist.png"), dpi=150)
    plt.close()

    # --- plot 4: corners over time ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, pred_vals, man_vals, title in zip(
        axes.flat,
        [pred_tl_x, pred_tl_y, pred_br_x, pred_br_y],
        [man_tl_x,  man_tl_y,  man_br_x,  man_br_y],
        ["Top-left X", "Top-left Y", "Bottom-right X", "Bottom-right Y"]
    ):
        ax.plot(frames, pred_vals, label="predicted")
        ax.plot(frames, man_vals,  label="manual", linestyle='--')
        ax.set_title(title); ax.set_xlabel("Frame")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corners.png"), dpi=150)
    plt.close()

    output = os.path.join(output_dir, f"statistics.txt")
    with open(output, 'w') as file:
        if runtime:
            file.write(f"Runtime: {runtime} secs\n")
            file.write(f"\n--- Evaluation over {len(common)} frames ---")
            file.write(f"  Mean IoU:            {np.mean(ious):.4f}")
            file.write(f"  Mean MSE:            {np.mean(mses):.4f}")
            file.write(f"  Mean center dist:    {np.mean(center_dists):.2f} px")

    # print summary
    print(f"\n--- Evaluation over {len(common)} frames ---")
    print(f"  Mean IoU:            {np.mean(ious):.4f}")
    print(f"  Mean MSE:            {np.mean(mses):.4f}")
    print(f"  Mean center dist:    {np.mean(center_dists):.2f} px")

def normalize(img):
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.astype(np.uint8)

    # grayscale -> 3 channel BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_rgb

def iter_frames(input_dir: str, poll_interval: float = 0.1, max_frames: int = None):
    seen = set()
    while True:
        available = natsorted(Path(input_dir).glob("*.npy"))
        new = [f for f in available if f not in seen]
        
        for f in new:
            seen.add(f)
            time.sleep(poll_interval)
            frame = np.load(f)
            frame = normalize(frame)
            yield frame
            if max_frames and len(seen) >= max_frames:
                return
        
        if not new:
            time.sleep(poll_interval)

#---- main function: inputs = file folder of numpy arrays ----------------------------------

if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="path to folder of .npy frames")
    parser.add_argument("init_box_dir", type=str, help="directory of npy file with initial box as x,y,w,h e.g. 316,385,742,488")
    parser.add_argument("--num_frames", type=int, help="number of total frames")
    parser.add_argument("--window", type=int, default=5, help="frames per batch (default 5)")
    parser.add_argument("--score_thresh", type=float, default=0)
    args = parser.parse_args()

    initial_box = initial_coords(args.init_box_dir)
    current_box = initial_box

    predictor = build_predictor()

    all_boxes = {}
    frame_buffer = []
    global_idx = 0

    start_time = datetime.now()

    for frame in iter_frames(input_dir=args.input_dir, max_frames=args.num_frames):
        frame_buffer.append(frame)

        if len(frame_buffer) < args.window:
            continue
    
        boxes = track_boundingbox(frame_buffer, current_box, predictor, args.score_thresh)

        all_boxes[global_idx] = boxes[0][1]
        current_box = boxes[1][1]

        global_idx += 1

        # move forward by one image
        frame_buffer.pop(0)

    end_time = datetime.now()
    runtime = end_time-start_time

    # visualization & statistics
    output_root = "./my_data_box" 
    dpath = Path(args.input_dir)
    output_full = output_root / Path(*dpath.parts[-2:])
    output_full.mkdir(parents=True, exist_ok=True)

    output_pts_dir = output_full / "points"
    output_img_dir = output_full / "box_images"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_stats_dir = output_full / "stats"
    output_stats_dir.mkdir(parents=True, exist_ok=True)

    save_boxcoords(all_boxes, output_pts_dir)

    manual_boxes = load_manual_boxes(args.init_box_dir)
    evaluate_and_plot(all_boxes, manual_boxes, output_dir=output_stats_dir)

    save_vid(all_boxes, args.input_dir, output_img_dir)
    motion_video(output_img_dir, output_stats_dir, name = None)
