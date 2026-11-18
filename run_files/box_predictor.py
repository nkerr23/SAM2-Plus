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

from run_files.box_inf import box_predictor, track_boundingbox
from run_files.tracking_results import TrackingResult

from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from tools.visualization import show_mask, show_box, show_points
from run_files.img_input import motion_video, normalize

#---- INPUT ------------------------------------------------------------------------------

'''
input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152'
manual_boundbox = '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152/liver'
output_root = "./my_data_box" 
'''

#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------

#---- helper functions ------------------------------------------------------------------

def initial_coords(coords_dir: str):
    files = natsorted(Path(coords_dir).glob("*.npy"))
    first_file = files[0]
    box = np.load(first_file)
    box_xyxy = np.load(first_file).astype(np.float32).reshape(4)
    x1, y1, x2, y2 = box_xyxy

    box_xywh = np.array([
        x1,
        y1,
        x2 - x1,
        y2 - y1
    ], dtype=np.float32)

    print("xyxy:", box_xyxy)
    print("xywh:", box_xywh)
    #print(box)
    #print(type(box))
    #print(box.dtype)
    #print(box.shape)
    #print(box)
    return box_xywh

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

def get_first_frame_shape(input_dir: str):
    files = natsorted(Path(input_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy frames found in {input_dir}")
    return np.load(files[0]).shape

def clamp_crop_roi(cropped_roi, frame_shape):
    if cropped_roi is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = map(int, cropped_roi)
    x1 = min(max(x + w, 0), frame_w)
    y1 = min(max(y + h, 0), frame_h)
    x0 = min(max(x, 0), frame_w)
    y0 = min(max(y, 0), frame_h)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Crop ROI {cropped_roi} is outside frame bounds "
            f"(width={frame_w}, height={frame_h})"
        )

    return [x0, y0, x1 - x0, y1 - y0]

def crop_frame(frame, cropped_roi):
    if cropped_roi is None:
        return frame
    x, y, w, h = clamp_crop_roi(cropped_roi, frame.shape)
    return frame[y:y + h, x:x + w, ...]

def load_all_frames(input_dir: str, cropped_roi=None):
    files = natsorted(Path(input_dir).glob("*.npy"))

    frames = []

    for f in files:
        frame = np.load(f)
        frame = normalize(frame)
        frame = crop_frame(frame, cropped_roi)
        frames.append(frame)

    return frames

def iter_frames(input_dir: str, poll_interval: float = 0.1, max_frames: int = None, cropped_roi=None):
    seen = set()
    while True:
        available = natsorted(Path(input_dir).glob("*.npy"))
        new = [f for f in available if f not in seen]
        
        for f in new:
            seen.add(f)
            time.sleep(poll_interval)
            frame = np.load(f)
            frame = normalize(frame)
            frame = crop_frame(frame, cropped_roi)
            yield frame
            if max_frames and len(seen) >= max_frames:
                return
        
        if not new:
            time.sleep(poll_interval)

def box_output(input_dir, init_box_dir, all_boxes):
    output_root = "./my_data_box" 
    dpath = Path(input_dir)
    output_full = output_root / Path(*dpath.parts[-2:])
    output_full.mkdir(parents=True, exist_ok=True)

    output_pts_dir = output_full / "points"
    output_img_dir = output_full / "box_images"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_stats_dir = output_full / "stats"
    output_stats_dir.mkdir(parents=True, exist_ok=True)

    save_boxcoords(all_boxes, output_pts_dir)

    manual_boxes = load_manual_boxes(init_box_dir)
    evaluate_and_plot(all_boxes, manual_boxes, output_dir=output_stats_dir)

    save_vid(all_boxes, input_dir, output_img_dir)
    motion_video(output_img_dir, output_stats_dir, name = None)

#---- main function: inputs = file folder of numpy arrays ----------------------------------

#python run_files/box_predictor.py '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152' '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152' --all_frames

if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="path to folder of .npy frames")
    parser.add_argument("init_box_dir", type=str, help="directory of npy file with initial box as x,y,w,h e.g. 316,385,742,488")
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--num_frames", type=int,help="number of total frames")
    parser.add_argument("--window", type=int, default=8, help="frames per batch (default 5)")
    parser.add_argument("--step", type=int, default=2, help="step size (default 3)")
    parser.add_argument("--score_thresh", type=float, default=0)
    parser.add_argument("--crop_roi", action=argparse.BooleanOptionalAction, default=True, help="crops ROI around initial bounding box to reduce computational time")
    parser.add_argument("--max_size_change", type=float, default=0.1, help="maximum fractional width/height change from the initial box (default: 0.1)")
    args = parser.parse_args()

    if args.window <= 0 or args.step <= 0 or args.step > args.window:
        parser.error("--window and --step must be positive, with --step <= --window")
    if not 0 <= args.max_size_change < 1:
        parser.error("--max_size_change must be in [0, 1)")

    initial_box = initial_coords(args.init_box_dir)
    current_box = initial_box
    box_size_anchor = initial_box[-2:].copy()
    box_predictor = box_predictor()
    all_boxes = {}

    cropped_roi = None
    if args.crop_roi:
        # crop images surrounding ROI [x_crop=x-x/2, y_crop=y-y/2, w+2x_crop, h+2y_crop]
        x, y, w, h = initial_box
        crop_margin = 0.30

        side = max(w, h) * (1 + 2 * crop_margin)

        center_x = x + w / 2
        center_y = y + h / 2

        x_crop = center_x - side / 2
        y_crop = center_y - side / 2
        w_crop = side
        h_crop = side
        x_crop, y_crop, w_crop, h_crop = map(int, (x_crop, y_crop, w_crop, h_crop))
        cropped_roi = clamp_crop_roi(
            [x_crop, y_crop, w_crop, h_crop],
            get_first_frame_shape(args.input_dir),
        )
        x_crop, y_crop, w_crop, h_crop = cropped_roi
        current_box = [x-x_crop, y-y_crop, w, h]

    out_idx = 0
    window_idx = 0
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    if args.all_frames:
        frames = load_all_frames(args.input_dir, cropped_roi=cropped_roi)
        all_boxes = track_boundingbox(
            frames, current_box, box_predictor, args.score_thresh,
            box_size_anchor=box_size_anchor,
            max_size_change=args.max_size_change,
        )
    else:
        
        frame_buffer = []
        global_idx = 0

        for frame in iter_frames(input_dir=args.input_dir, max_frames=args.num_frames, cropped_roi=cropped_roi):
            frame_buffer.append(frame)

            if len(frame_buffer) < args.window:
                continue 

            boxes = track_boundingbox(
                frame_buffer, current_box, box_predictor, args.score_thresh,
                box_size_anchor=box_size_anchor,
                max_size_change=args.max_size_change,
            )

            if window_idx == 0:
                # first window: save frames 0,1,2,3,4
                for i in range(len(boxes)):
                    all_boxes[out_idx] = boxes[i]#[1]
                    out_idx += 1
            else:
                # later windows: skip frame 0 because it overlaps with previous window
                for i in range(args.step, len(boxes)):
                    all_boxes[out_idx] = boxes[i]#[1]
                    out_idx += 1

            print(type(all_boxes[0]))
            print(all_boxes[0])

            #all_boxes[global_idx] = boxes[0][1]
            #current_box = boxes[1][1]
            sorted_keys = sorted(boxes)
            prompt_key = sorted_keys[-args.step]
            current_box = boxes[prompt_key][1].copy()

            window_idx += 1
            global_idx += 1

            # move forward by one image
            #frame_buffer.pop(0)
            frame_buffer = frame_buffer[-args.step:]

    if args.crop_roi:
        for frame_idx, frame_boxes in all_boxes.items():
            for object_id, box_array in frame_boxes.items():
                box_array[..., 0] += x_crop
                box_array[..., 1] += y_crop

    torch.cuda.synchronize()
    runtime = time.perf_counter() - start_time
    #runtime = end_time-start_time

    # visualization & statistics
    #box_output(args.input_dir, args.init_box_dir, all_boxes)

    result = TrackingResult(
        granularity='box', 
        input_dir=args.input_dir,
        output_root="./sam2_output_rt" ,
        results=all_boxes,
        runtime=runtime
    )
    
    result.save_output()
    result.load_reference(args.init_box_dir)
    result.draw_box_overlay()
    result.box_center()
    result.motion_video(result.output_overlay, name = 'reference')
    result.motion_video(result.output_compare, name = 'comparison')
    result.print_stats(result.resu_center, result.ref_center, 'boxcenter_comparison')
    result.graph_2traj(result.resu_center, result.ref_center, name = 'boxcenter_comparison_graph', labels=['SAM 2++ Tracking', 'Manual Tracking'])
