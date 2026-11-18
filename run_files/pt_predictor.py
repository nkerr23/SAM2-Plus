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

from run_files.point_inf import point_predictor, track_keypoint
from run_files.tracking_results import TrackingResult

from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from tools.visualization import show_mask, show_box, show_points
from run_files.img_input import normalize, Kalman2D

#---- INPUT ------------------------------------------------------------------------------

'''
input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152'
manual_boundbox = '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152/liver'
output_root = "./my_data_box"
'''

#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------

#---- helper functions ------------------------------------------------------------------

def load_points(points_dir: str):
    subfolders = natsorted([
        f for f in Path(points_dir).iterdir()
        if f.is_dir()
    ])

    all_points = []
    for folder in subfolders:
        files = natsorted(folder.glob("*.npy"))
        point_traj = []

        for f in files:
            pt = np.load(f).astype(np.float32).reshape(2)
            point_traj.append(pt)

        # shape: (num_frames, 2)
        point_traj = np.stack(point_traj, axis=0)
        all_points.append(point_traj)

    # shape: (num_points, num_frames, 2)
    all_points = np.stack(all_points, axis=0)
    # convert to (num_frames, num_points, 2)
    trajs_2d = np.transpose(all_points, (1, 0, 2))

    #num_frames, num_points, _ = trajs_2d.shape
    #visibs = np.ones((num_frames, num_points), dtype=bool)

    trajs_2d = np.zeros((num_frames, num_points, 2), dtype=np.float32)
    visibs = np.zeros((num_frames, num_points), dtype=bool)

    trajs_2d[0, :, :] = initial_points
    visibs[0, :] = True
    

    return {
        "trajs_2d": trajs_2d,
        "visibs": visibs
    }

def load_initial_points(points_dir: str):
    subfolders = natsorted([
        f for f in Path(points_dir).iterdir()
        if f.is_dir()
    ])

    initial_points = []

    for folder in subfolders:
        files = natsorted(folder.glob("*.npy"))
        pt = np.load(files[0])
        num_frames = len(files)
        pt = np.asarray(pt, dtype=np.float32).reshape(2)
        initial_points.append(pt)

    initial_points = np.stack(initial_points, axis=0)
      # (num_points, 2)
    num_points = initial_points.shape[0]

    trajs_2d = np.zeros((1, num_points, 2), dtype=np.float32)
    visibs = np.zeros((1, num_points), dtype=bool)


    #trajs_2d = np.zeros((num_frames, num_points, 2), dtype=np.float32)
    #visibs = np.zeros((num_frames, num_points), dtype=bool)

    trajs_2d[0] = initial_points
    visibs[0, :] = True

    return {
        "trajs_2d": trajs_2d,
        "visibs": visibs
    }

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

def square_crop_for_points(points, frame_shape, margin=0.3, min_size=256):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    frame_h, frame_w = frame_shape[:2]
    if points.size == 0:
        raise ValueError("At least one initial point is required for cropping")
    if not np.isfinite(points).all():
        raise ValueError("Initial points must contain only finite coordinates")
    if margin < 0:
        raise ValueError("Crop margin must be non-negative")
    if min_size <= 0:
        raise ValueError("Minimum crop size must be positive")
    if (
        np.any(points[:, 0] < 0) or np.any(points[:, 0] >= frame_w)
        or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= frame_h)
    ):
        raise ValueError("Initial points must be inside the image")

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2
    point_extent = float(np.max(max_xy - min_xy))
    side = int(np.ceil(max(min_size, point_extent * (1 + 2 * margin))))
    side = min(side, frame_w, frame_h)

    if np.any(max_xy - min_xy > side):
        raise ValueError("A square crop cannot contain all points in this frame")

    x = int(np.clip(np.floor(center[0] - side / 2), 0, frame_w - side))
    y = int(np.clip(np.floor(center[1] - side / 2), 0, frame_h - side))
    if np.any(points[:, 0] < x) or np.any(points[:, 0] >= x + side):
        raise ValueError("Crop does not contain every initial point in x")
    if np.any(points[:, 1] < y) or np.any(points[:, 1] >= y + side):
        raise ValueError("Crop does not contain every initial point in y")
    return [x, y, side, side]

def crop_frame(frame, cropped_roi):
    if cropped_roi is None:
        return frame
    x, y, w, h = cropped_roi
    return frame[y:y + h, x:x + w, ...]

def load_all_frames(input_dir: str):
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

def img():
    os.makedirs(output_point_dir, exist_ok=True)
    np.savez(os.path.join(output_point_dir, f"{video_name}.npz"), trajs_2d=point_array, visibs=visible_array, size=(width, height))

#---- main function: inputs = file folder of numpy arrays ----------------------------------
#python run_files/pt_predictor.py '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152' '../data/manual_tracking_coords/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152' --all_frames

if __name__ == "__main__":
    #command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="path to folder of .npy frames")
    parser.add_argument("init_point_dir", type=str, help="directory of npy file with initial point(s) as x,y")
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--num_frames", type=int, help="number of total frames")
    parser.add_argument("--window", type=int, default=8, help="frames per batch (default 5)")
    parser.add_argument("--step", type=int, default=2, help="step size (default 3)")
    parser.add_argument("--score_thresh", type=float, default=0)
    parser.add_argument("--crop_roi", action=argparse.BooleanOptionalAction, default=True, help="crop a square ROI around all initial points")
    parser.add_argument("--crop_margin", type=float, default=0.3, help="fractional margin around the full initial-point extent (default: 0.3)")
    parser.add_argument("--min_crop_size", type=int, default=256, help="minimum square crop side in source-image pixels (default: 256)")
    args = parser.parse_args()

    initial_points = load_initial_points(args.init_point_dir)

    cropped_roi = None
    if args.crop_roi:
        cropped_roi = square_crop_for_points(
            initial_points["trajs_2d"][0],
            get_first_frame_shape(args.input_dir),
            margin=args.crop_margin,
            min_size=args.min_crop_size,
        )
        x_crop, y_crop, _, _ = cropped_roi
        initial_points["trajs_2d"][..., 0] -= x_crop
        initial_points["trajs_2d"][..., 1] -= y_crop

    current_points = initial_points
    print(current_points["trajs_2d"].shape)
    print(current_points["visibs"].shape)
    point_predictor = point_predictor()
    all_points = []

    num_points = current_points["trajs_2d"].shape[1]
    kalman_filters = {i: Kalman2D() for i in range(num_points)}
    alpha = 0.5
    allframes_firstwindow = True

    #start_time = datetime.now()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    if args.all_frames:
        frames = load_all_frames(args.input_dir, cropped_roi=cropped_roi)
        all_points, _, _, _ = track_keypoint(frames, current_points, point_predictor, args.score_thresh, kalman_filters, allframes_firstwindow)
    else:
        frame_buffer = []
        global_idx = 0

        for frame in iter_frames(input_dir=args.input_dir, max_frames=args.num_frames, cropped_roi=cropped_roi):
            frame_buffer.append(frame)

            #frame buffer must be size of window to continue, otherwise keep collecting
            if len(frame_buffer) < args.window:
                continue

            print("context being passed to track_keypoint:")
            print(current_points['trajs_2d'])  # should show 4 different positions, not zeros   

            #track points
            points, visible_array, width, height = track_keypoint(frame_buffer, current_points, point_predictor, args.score_thresh, kalman_filters, allframes_firstwindow)

            allframes_firstwindow = False
            
            #saving frames
            if global_idx == 0:
                # first window: save all frames
                #all_points.extend(points)
                for f_idx in range(points.shape[0]):
                    all_points.append(points[f_idx])
            else:
                # later windows: save only newest frame
                #all_points.append(points[-1])
                all_points.extend(points[args.step:])

            global_idx += 1

            # move forward by one image
            #frame_buffer.pop(0)

            # FIX: Slice directly from the fresh 'points' matrix generated in this iteration,
            # not from the accumulated global history.
            # If the current window was [F3, F4, F5, F6, F7], 'points' contains [C3, C4, C5, C6, C7].
            # For the next window [F4, F5, F6, F7, F8], we want to pass [C4, C5, C6, C7].
            context_size = args.window - 1
            recent_trajectories = points[-context_size:] 
            
            current_points = {
                "trajs_2d": points[-args.step:].copy(),
                "visibs": visible_array[-args.step:].copy()
            }

            #frame_buffer.pop(0)
            frame_buffer = frame_buffer[-args.step:]

    #end_time = datetime.now()
    #runtime = end_time-start_time
    torch.cuda.synchronize()
    runtime = time.perf_counter() - start_time

    all_points = np.asarray(all_points, dtype=np.float32)
    if cropped_roi is not None:
        all_points[..., 0] += x_crop
        all_points[..., 1] += y_crop

    print(all_points)

    # visualization & statistics
    result = TrackingResult(
        granularity='point', 
        input_dir=args.input_dir,
        output_root="./sam2_output_rt" ,
        results=all_points,
        runtime=runtime
    )
    
    result.load_ref_multipt(args.init_point_dir)
    result.save_output()
    result.draw_points_overlay()
    result.motion_video(result.output_overlay, name = 'reference')
    result.motion_video(result.output_compare, name = 'comparison')
    result.print_stats(result.results[:, 0, :], result.reference[:, 0, :], 'boxcenter_comparison', length=args.num_frames)
    result.plot_multi_motion()
    