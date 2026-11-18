import numpy as np
import cv2
import os
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from natsort import natsorted
from datetime import datetime
import math
import matplotlib.pyplot as plt

from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from run_files.img_input import normalize

class TrackingResult:
    def __init__(self, granularity, input_dir, output_root, results, runtime):
        self.input_dir = input_dir
        self.output_root = output_root
        self.runtime = runtime
        self.granularity = granularity
        self.interp_scale = 4

        self.colors_rgb = [
            (0, 255, 255),      # cyan
            (255, 165, 0),      # orange
            (160, 32, 240),      # purple
            (255, 105, 180),    # pink
            (0, 120, 255),      # blue
            (0, 255, 0)
        ]
        self.colors_bgr = [
            (255, 255, 0),      # cyan
            (0, 165, 255),      # orange
            (240, 32, 160),      # purple
            (180, 105, 255),    # pink
            (255, 120, 0),      # blue
            (0, 255, 0)
        ]

        if granularity == 'box':
            converted_results = {}
            converted_results_interp = {}
            for frame_idx, boxes_xywh in results.items():
                obj_dict = results[frame_idx]
                # grab first object's box
                boxes_xywh = np.array(list(obj_dict.values()), dtype=np.float32).reshape(-1, 4)
                boxes_xyxy = boxes_xywh.copy()
                boxes_xyxy[:,2] += boxes_xyxy[:,0]
                boxes_xyxy[:,3] += boxes_xyxy[:,1]
                converted_results[frame_idx] = boxes_xyxy.astype(int).reshape(-1)
            self.results = np.stack([converted_results[k] for k in sorted(converted_results.keys())],axis=0)
        elif granularity == 'point':
            self.results = np.asarray(results, dtype=np.float32)
        else:
            self.results = results

        self.reference = None
        self.frames = None
        self.output = None
        self.ref_center = None
        self.resu_center = None
        self.stats = {}
        self.pt_folders = None

        self.output_stats = None
        self.output_pts = None
        self.output_ref = None
        self.output_img = None
        self.output_overlay = None
        self.output_compare = None

        self.load_dirs()
        self.load_frames()

    #load directories
    def load_dirs(self):
        output_root = self.output_root
        input_dir = self.input_dir

        dpath = Path(input_dir)
        output_dir = output_root / Path(*dpath.parts[-2:]) / self.granularity
        output_dir.mkdir(parents=True, exist_ok=True)

        def make_dir(name):
            output_path = output_dir / name
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
        
        self.output_stats = make_dir('stats')
        self.output_img = make_dir('images')
        self.output_overlay = make_dir('overlay')
        self.output_compare = make_dir('manual_overlay')
        self.output_pts = make_dir('output')
        self.output_ref = make_dir('reference_annotation')

    #load numpy frames to var & save to output & interpolate up by 4
    def load_frames(self):
        files = natsorted(Path(self.input_dir).glob("*.npy"))
        frames = [np.load(f) for f in files]
        frames = np.stack(frames, axis=0)   
        frames = frames[:len(self.results)]

        upsampled_frames = []
        for i, frame in enumerate(frames):
            # interpolate by factor of 4
            up_frame = cv2.resize(frame, None, fx=self.interp_scale, fy=self.interp_scale, interpolation=cv2.INTER_CUBIC)

            upsampled_frames.append(up_frame)
            
            np.save(os.path.join(self.output_img, f"{i:04d}.npy"), up_frame)

        self.frames = np.stack(upsampled_frames, axis=0)
        #for idx, box in self.results.items():
        #    np.save(os.path.join(self.output_img, f"{idx:04d}.npy"), box)
        #print(f"Saved {len(self.results)} boxes to {self.output_img}")

    #maybe edit for points with multi point files
    #load manually annotated reference data & put in numpy & scale by 4
    def load_reference(self, manual_dir):
        files = natsorted(Path(manual_dir).glob("*.npy"))
        self.reference = np.stack([np.load(f).astype(float) for f in files], axis=0)
        np.save(os.path.join(self.output_ref, "reference.npy"), self.reference)

    #load manually annotated reference points (multiple)
    def load_ref_multipt(self, manual_dir):
        point_folders = [
            d for d in os.listdir(manual_dir)
            if os.path.isdir(os.path.join(manual_dir, d))
        ]
        point_folders = natsorted(point_folders)

        if len(point_folders) == 0:
            raise ValueError(f"No point folders found in {point_tracking_root}")

        all_points = []
        expected_num_frames = None

        for point_name in point_folders:
            point_dir = os.path.join(manual_dir, point_name)

            coord_files = [
                f for f in os.listdir(point_dir)
                if f.endswith(".npy")
            ]
            coord_files = natsorted(coord_files)

            point_coords = []
            point_coords_interp = []

            for fname in coord_files:
                fpath = os.path.join(point_dir, fname)
                arr = np.load(fpath)
                arr = np.asarray(arr, dtype=np.float32)
                arr_interp = np.asarray(arr, dtype=np.float32) * self.interp_scale

                if arr.ndim == 2 and arr.shape == (1, 2):
                    arr = arr[0]
                    arr_interp = arr_interp[0]

                if arr.shape != (2,):
                    raise ValueError(f"{fpath} has shape {arr.shape}, expected (2,)")

                point_coords.append(arr)
                point_coords_interp.append(arr_interp)

            point_coords = np.stack(point_coords, axis=0)  # (num_frames, 2)
            point_coords_interp = np.stack(point_coords_interp, axis=0)

            if expected_num_frames is None:
                expected_num_frames = point_coords.shape[0]
            elif point_coords.shape[0] != expected_num_frames:
                raise ValueError(
                    f"{point_dir} has {point_coords.shape[0]} frames, expected {expected_num_frames}"
                )

            np.save(os.path.join(self.output_ref, f"{point_name}.npy"), all_points)
            all_points.append(point_coords)

        nested_array = np.stack(all_points, axis=1)  # (num_frames, num_points, 2)

        self.pt_folders = point_folders
        self.reference = nested_array

    #load result data to numpy
    def save_output(self):
        npy_arr = self.results
        if self.granularity == 'point':
            for i, point_name in enumerate(self.pt_folders):
                np.save(os.path.join(self.output_pts, f"{point_name}.npy"), npy_arr[:, i, :])
        else:
            np.save(os.path.join(self.output_pts, "results.npy"), self.results)
        
    #draw box overlay
    def draw_box_overlay(self):
        for i, img in enumerate(self.frames):
            # load and normalize frame
            frame = normalize(img)
            if i == 1:
                print(frame.min(), frame.max(), frame.dtype, frame.shape)

            color = self.colors_bgr[-1]
            color_ref = (0, 0, 255)

            #sam2++ output prediction
            x1, y1, x2, y2 = self.results[i].astype(int) * self.interp_scale
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=self.interp_scale)
            cv2.imwrite(os.path.join(self.output_overlay, f"{i:05d}_withbox.jpg"),frame)

            #manual & sam2 output comparison
            mx1, my1, mx2, my2 = self.reference[i].astype(int) * self.interp_scale
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color=color_ref, thickness=self.interp_scale)
            cv2.imwrite(os.path.join(self.output_compare, f"{i:05d}_compare.jpg"),frame)

        print(f"Saved {len(self.frames)} frames to {self.output_overlay} and {self.output_compare}")

    #draw points overlay
    def draw_points_overlay(self):
        n = min(len(self.frames), (self.reference).shape[0])
        radius = 2
        thick = -1

        for i in range(n):
            frame = normalize(self.frames[i])
            frame_compare = frame.copy()

            ref_points = self.reference[i] * self.interp_scale  # shape: (num_points, 2)
            resu_points = self.results[i] * self.interp_scale

            for j, point in enumerate(resu_points):
                x, y = point
                x, y = int(round(x)), int(round(y))
                x1, y1 = ref_points[j]
                x1, y1 = int(round(x1)), int(round(y1))

                color = self.colors_bgr[j]
                color_ref = (0, 0, 255)

                cv2.circle(frame, (x, y), int(2 * self.interp_scale), color=color, thickness=thick)

                cv2.circle(frame_compare, (x1, y1), int(2 * self.interp_scale), color=color_ref, thickness=thick)
                cv2.circle(frame_compare, (x, y), int(2 * self.interp_scale), color=color, thickness=thick)
            
            cv2.imwrite(os.path.join(self.output_compare, f"{i:05d}_withpoints.jpg"),frame_compare)
            cv2.imwrite(os.path.join(self.output_overlay, f"{i:05d}_withpoints.jpg"),frame)
        
        print(f"Saved {n} point overlay frames to {self.output_overlay}")

    #graph box center
    def box_center(self):
        reference_center = []
        result_center = []

        for i, box in enumerate(self.results):
            x1, y1, x2, y2 = box
            mx1, my1, mx2, my2 = self.reference[i]
            result_center.append([((x1+x2)/2), ((y1+y2)/2)])
            reference_center.append([((mx1+mx2)/2), ((my1+my2)/2)])

        self.resu_center = np.array(result_center)
        self.ref_center = np.array(reference_center)

    #return frames in video
    def motion_video(self, input_folder, name = None):
        output_path = self.output_stats

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

    #print comparison statistics
    def print_stats(self, arr1, arr2, name = None, length=None):
        x0 = arr1[:, 0]
        y0 = arr1[:, 1]
        x1 = arr2[:, 0]
        y1 = arr2[:, 1]

        if length != None:
            x1 = x1[:length]
            y1 = y1[:length]

        mse_value_y = (mean_squared_error(y0, y1))**0.5
        mse_value_x = (mean_squared_error(x0, x1))**0.5

        x_cc, _ = pearsonr(x0, x1)
        y_cc, _ = pearsonr(y0, y1)
        np.round(x_cc, 3)
        np.round(y_cc, 3)

        avg_time = self.runtime / len(x0)

        if (name):
            output = os.path.join(self.output_stats, f"{name}.txt")
        else:
            output = os.path.join(self.output_stats, f"statistics.txt")
        with open(output, 'w') as file:
            #file.write(f"{folder}\n\n")
            file.write(f"Root mean squared error:\n\ty: {mse_value_y}\n\tx: {mse_value_x}\n\n")
            file.write(f"Correlation coefficient:\n\ty corr coeff: {y_cc}\n\n")
            file.write(f"\tx corr coeff: {x_cc}\n")
            file.write(f"Runtime: {self.runtime} secs\n")
            file.write(f"Runtime per frame: {avg_time} secs")

    def print_pt_stats(self, arr1, arr2, name = None):
        x0 = arr1[:, 0]
        y0 = arr1[:, 1]
        x1 = arr2[:, 0]
        y1 = arr2[:, 1]

        mse_value_y = (mean_squared_error(y0, y1))**0.5
        mse_value_x = (mean_squared_error(x0, x1))**0.5

        x_cc, _ = pearsonr(x0, x1)
        y_cc, _ = pearsonr(y0, y1)
        np.round(x_cc, 3)
        np.round(y_cc, 3)

        avg_time = self.runtime / len(x0)

        if (name):
            output = os.path.join(self.output_stats, f"{name}.txt")
        else:
            output = os.path.join(self.output_stats, f"statistics.txt")
        with open(output, 'w') as file:
            #file.write(f"{folder}\n\n")
            file.write(f"Root mean squared error:\n\ty: {mse_value_y}\n\tx: {mse_value_x}\n\n")
            file.write(f"Correlation coefficient:\n\ty corr coeff: {y_cc}\n\n")
            file.write(f"\tx corr coeff: {x_cc}\n")
            file.write(f"Runtime: {self.runtime} secs\n")
            file.write(f"Runtime per frame: {avg_time} secs")

    def plot_all():

        dt = 6.31*16/1000 # time resolution in second 
        dy = 1.5625 # spatial resolution
        dx = 1.5625 # spatial resolution
        time = np.arange(len(x0)) * dt

    #graph x and y comparison of 2 trajectories
    def graph_2traj(self, arr1, arr2, name = None, labels=['SAM 2++ Tracking', 'Manual Tracking'], title='Predicted vs. Reference Motion Over Time'):

        x0 = arr1[:, 0]
        y0 = arr1[:, 1]
        x1 = arr2[:, 0]
        y1 = arr2[:, 1]

        dt = 6.31*16/1000 # time resolution in second 
        dy = 1.5625 # spatial resolution
        dx = 1.5625 # spatial resolution
        time = np.arange(len(x0)) * dt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

        dxx_mm0 = np.array(x0) * dx
        dyy_mm0 = np.array(y0) * dy
        dxx_mm1 = np.array(x1) * dx
        dyy_mm1 = np.array(y1) * dy

        color_ref = (1, 0, 0)
        color = tuple(c / 255 for c in self.colors_rgb[-1])

        ax1.plot(time, dxx_mm0, label=labels[0], color=color)
        ax1.plot(time, dxx_mm1, label=labels[1], color=color_ref)
        ax1.set_title(title)
        ax1.set_ylabel('x Position (mm)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time, dyy_mm0, label=labels[0], color=color)
        ax2.plot(time, dyy_mm1, label=labels[1], color=color_ref)
        ax2.set_ylabel('y Position (mm)')
        ax2.set_xlabel('Time (s)')
        ax2.legend()
        ax2.grid(True)

        #get both graphs to have same y-axes scale
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        range1 = ylim1[1] - ylim1[0]
        range2 = ylim2[1] - ylim2[0]
        target = max(range1, range2)
        center1 = (ylim1[0] + ylim1[1]) / 2
        center2 = (ylim2[0] + ylim2[1]) / 2

        ax1.set_ylim(center1 - target/2, center1 + target/2)
        ax2.set_ylim(center2 - target/2, center2 + target/2)

        plt.tight_layout()
        if (name):
            output = os.path.join(self.output_stats, f"{name}")
        else:
            output = os.path.join(self.output_stats, f"2traj_comparison")
        plt.savefig(output, dpi=300)

    def plot_multi_motion(self):
        dt=6.31*16/1000
        dx=1.5625
        dy=1.5625

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(8, 6),
            sharex=True
        )

        point_arr = self.results

        num_frames, num_points, _ = point_arr.shape

        for i in range(num_points):
            coords = point_arr[:, i, :]
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

            color = tuple(c / 255 for c in self.colors_rgb[i])
            label = self.pt_folders[i] if i < len(self.pt_folders) else f"Point {i}"

            ax1.plot(time, dx_mm, linewidth=2, label=label, color=color)
            ax2.plot(time, dy_mm, linewidth=2, label=label, color=color)

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

        output_path = self.output_stats / 'all_pts_graph'

        if output_path is not None:
            plt.savefig(output_path, dpi=300)
            print(f"Saved plot to {output_path}")
        else:
            plt.show()    

