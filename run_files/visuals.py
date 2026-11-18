import os
import cv2
import sys
import numpy as np
from pathlib import Path
from natsort import natsorted
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import math

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from run_files.tracking_results import TrackingResult

#0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152
#0223_RTimages/CV_GARADIAL_GRE1_OBL_SAG_3SL_DYNAMIC_0305
#0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0347
input_dir = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0154'

interp_scale = 4

dpath = Path(input_dir)
img_path = Path(*dpath.parts[-2:]) 

box_npy = Path('../SAM2-Plus/sam2_output_rt') / img_path / 'box' / 'output' / 'results.npy'
output_base = Path('../SAM2-Plus/sam2_output_rt') / img_path
point_img_dir =  output_base / 'point' / 'overlay'
output_root = 'visuals'

output_dir = output_base / 'visuals' #output
output_dir.mkdir(parents=True, exist_ok=True)
output_pt_box = output_dir / 'box_point' #input_path
output_pt_box.mkdir(parents=True, exist_ok=True)

image = output_pt_box / '00000_withpoints.jpg'
output_err = output_dir / 'error_bound.jpg'

point_folder = output_base / 'point' / 'output'
output_img_pt = output_dir / 'pt_graph'
output_img_all = output_dir / 'box_pt_graph'

colors_rgb = [
    (0, 255, 255),      # cyan
    (255, 165, 0),      # orange
    (160, 32, 240),      # purple
    (255, 105, 180),    # pink
    (0, 120, 255),      # blue
    (0, 255, 0)
]
colors_bgr = [
    (255, 255, 0),      # cyan
    (0, 165, 255),      # orange
    (240, 32, 160),      # purple
    (180, 105, 255),    # pink
    (255, 120, 0),      # blue
    (0, 255, 0)
]

def load_boxes(box_file):
    boxes = np.load(box_file, allow_pickle=True)
    boxes = np.asarray(boxes, dtype=np.float32)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"Expected shape (num_frames, 4), got {boxes.shape}"
        )

    return boxes # (num_frames, 4)

def draw_boxes_on_point_jpgs(point_img_dir, box_dir, output_dir, color=colors_bgr[-1], thickness=interp_scale):
    point_img_dir = Path(point_img_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = natsorted([
        f for f in point_img_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(img_files) == 0:
        raise ValueError(f"No image files found in {point_img_dir}")

    boxes = load_boxes(box_dir)

    n = min(len(img_files), len(boxes))

    for i in range(n):
        img = cv2.imread(str(img_files[i]))

        if img is None:
            print(f"Skipping unreadable image: {img_files[i]}")
            continue

        x1, y1, x2, y2 = boxes[i].astype(int)  * interp_scale
        x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)

        out_path = output_dir / img_files[i].name
        cv2.imwrite(str(out_path), img)

    print(f"Saved {n} images with points + boxes to {output_dir}")

def draw_points_and_box_margin(
    image,
    points,
    box,
    output_path,
    point_radius_mm=5,
    box_margin_mm=5,
    dx=1.5625,
    dy=1.5625,
):
    img = cv2.imread(str(image))
    if img is None:
        raise ValueError(f"Could not read image: {image}")
    frame = img.copy()

    draw_shift = 4
    fixed_scale = 1 << draw_shift
    point_radius_x = max(
        fixed_scale,
        round(point_radius_mm / dx * interp_scale * fixed_scale),
    )
    point_radius_y = max(
        fixed_scale,
        round(point_radius_mm / dy * interp_scale * fixed_scale),
    )
    margin_x = int(round(box_margin_mm / dx * interp_scale))
    margin_y = int(round(box_margin_mm / dy * interp_scale))
    center_dot_radius = max(1, round(interp_scale * 0.5))

    points_folder = Path(points)
    files = natsorted(points_folder.glob("*.npy"))

    all_points = []

    for f in files:
        coords = np.load(f).astype(np.float32)   # (num_frames, 2)
        first_point = coords[0]                  # first frame only
        all_points.append(first_point)

    points = np.stack(all_points, axis=0)
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        x *= interp_scale
        y *= interp_scale
        cv2.ellipse(
            frame,
            (x * fixed_scale, y * fixed_scale),
            (point_radius_x, point_radius_y),
            0,
            0,
            360,
            (0, 0, 255),
            2,
            shift=draw_shift,
        )
        cv2.circle(frame, (x, y), center_dot_radius, (255, 255, 255), -1)

    # original box
    box = np.load(box)
    box = box[0]
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32).reshape(4) * interp_scale
    x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])

    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # +5 mm outer box
    outer_x1 = x1 - margin_x
    outer_y1 = y1 - margin_y
    outer_x2 = x2 + margin_x
    outer_y2 = y2 + margin_y

    cv2.rectangle(frame, (outer_x1, outer_y1), (outer_x2, outer_y2), (0, 0, 255), interp_scale)

    # -5 mm inner box
    inner_x1 = x1 + margin_x
    inner_y1 = y1 + margin_y
    inner_x2 = x2 - margin_x
    inner_y2 = y2 - margin_y

    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        cv2.rectangle(frame, (inner_x1, inner_y1), (inner_x2, inner_y2), (0, 0, 255), interp_scale)

    success = cv2.imwrite(str(output_path), frame)
    print('imwrite success:', success)
    print('written to', output_path)

def plot_box_point_motion(
    points_folder,
    box_path,
    output_path=None,
    dt=6.31*16/1000,
    dx=1.5625,
    dy=1.5625,
):
    points_folder = Path(points_folder)

    files = natsorted(points_folder.glob("*.npy"))

    if len(files) == 0:
        raise ValueError(f"No .npy files found in {points_folder}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 6),
        sharex=True
    )

    for i, f in enumerate(files):
        coords = np.load(f).astype(np.float32)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"{f} has shape {coords.shape}, expected (num_frames, 2)"
            )

        x = coords[:, 0]
        y = coords[:, 1]

        # displacement from first frame in mm
        dx_mm = (x - x[0]) * dx
        dy_mm = (y - y[0]) * dy

        time = np.arange(len(coords)) * dt
        color = tuple(c / 255 for c in colors_rgb[i % len(colors_rgb)])

        ax1.plot(time, dx_mm, linewidth=2, label=f.stem, color=color)
        ax2.plot(time, dy_mm, linewidth=2, label=f.stem, color=color)

    boxes = np.load(box_path).astype(np.float32)  # (num_frames, 4), xyxy
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{box_path} has shape {boxes.shape}, expected (num_frames, 4)")

    box_cx = (boxes[:, 0] + boxes[:, 2]) / 2
    box_cy = (boxes[:, 1] + boxes[:, 3]) / 2

    box_dx_mm = (box_cx - box_cx[0]) * dx
    box_dy_mm = (box_cy - box_cy[0]) * dy
    box_time = np.arange(len(boxes)) * dt

    color = tuple(c / 255 for c in colors_rgb[-1])

    ax1.plot(box_time, box_dx_mm, linewidth=3, label="Box center", color=color)
    ax2.plot(box_time, box_dy_mm, linewidth=3, label="Box center", color=color)

    ax1.set_title("X Motion Over Time")
    ax1.set_ylabel("X Displacement (mm)")
    ax1.grid(True)

    ax2.set_title("Y Motion Over Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y Displacement (mm)")
    ax2.grid(True)

    # same y-scale on both plots
    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()

    min_y = min(ylim1[0], ylim2[0])
    max_y = max(ylim1[1], ylim2[1])

    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)

    ax1.legend(loc="upper left",bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def motion_video(input_folder, output_path, name = None):
        cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not name:
            name = cur_time  

        os.makedirs(output_path, exist_ok=True)
        path = os.path.join(output_path, f"{name}.avi")

        img_frames = natsorted([
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        first = cv2.imread(img_frames[0])
        if first is None:
            raise ValueError(f"Could not read first frame: {img_frames[0]}")
        height, width = first.shape[:2]

        dt = 6.31*16/1000 # time resolution in second 
        fps = math.ceil(1/dt)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        #height, width = img_frames[0].shape[:2]
        output = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame in img_frames:
            img = cv2.imread(frame)
            output.write(img)
        #print(cv2.getBuildInformation())
        output.release()
        print(f"Video saved to: {path}")

draw_boxes_on_point_jpgs(point_img_dir, box_npy, output_pt_box)
draw_points_and_box_margin(image=image, points=point_folder, box=box_npy, output_path=str(output_err))
motion_video(output_pt_box, output_dir, name='box_vs_points')
plot_box_point_motion(point_folder, box_npy, output_path=output_img_all)
