import os
import numpy as np
import cv2
import pydicom
from pathlib import Path
from datetime import datetime
import math
from natsort import natsorted
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def read_dicom_img(img_path, img_start_index):
    ima_files = []

    for i, file_name in enumerate(natsorted(os.listdir(img_path))):
        if (i < img_start_index - 1):
            continue

        file_path = os.path.join(img_path, file_name)
        ds = pydicom.dcmread(file_path)
        ima_files.append(ds)

    imgarray = np.stack([ds.pixel_array for ds in ima_files])
    imgarray = np.array([cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') for img in imgarray])
    
    imgarray = np.squeeze(imgarray)
    imgarray = imgarray.transpose(2,1,0)
    imgarray = np.flip(imgarray, 1)
    imgarray = np.rot90(imgarray, 1)

    return imgarray, imgarray.shape[2]

def dicom_to_jpg(input_path, output_root, start):
    [imgarr, dim] = read_dicom_img(input_path, start)

    dpath = Path(input_path)
    output_path = output_root / Path(*dpath.parts[-2:]) / 'images'
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for img_i in range(dim):
        img = imgarr[:,:,img_i]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        filename = os.path.join(output_path, f"{img_i:05d}.jpg")

        cv2.imwrite(filename, img_rgb)

    return str(output_path)

def numpy_to_npz(points_np, dicom_path, output_root):
    dpath = Path(dicom_path)
    npath = Path(points_np)
    output_path = output_root / Path(*dpath.parts[-2:]) / f"points"
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_dir = output_path / f"{npath.name}.npz"
    
    coords = []
    
    npy_files = natsorted(npath.glob("*.npy"))
    for f in natsorted(npy_files):
        pt = np.load(f).astype(np.float32)  # should be (2,)
        
        # Ensure correct shape
        pt = np.asarray(pt).reshape(1, 2)  # (1,2)
        coords.append(pt)

    coords = np.stack(coords, axis=0)  # (T,1,2)

    T, N, _ = coords.shape

    visibs = np.ones((T, N), dtype=bool)

    np.savez(
        output_dir,
        trajs_2d=coords,
        visibs=visibs
    )

    return str(output_dir)

def numpy_to_txt(points_np, dicom_path, output_root):
    dpath = Path(dicom_path)
    npath = Path(points_np)

    # same folder structure as numpy_to_npz
    output_path = output_root / Path(*dpath.parts[-2:]) / "points"
    output_path.mkdir(parents=True, exist_ok=True)

    # same naming pattern, just .txt instead of .npz
    output_file = output_path / f"{npath.name}.txt"

    npy_files = natsorted(npath.glob("*.npy"))

    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {npath}")

    with open(output_file, "w") as f_out:
        for f in npy_files:
            arr = np.load(f).flatten()

            if arr.shape != (4,):
                raise ValueError(f"{f} has shape {arr.shape}, expected (4,)")

            x1, y1, x2, y2 = arr
            f_out.write(f"{x1},{y1},{abs(x2-x1)},{abs(y2-y1)}\n")

    print(f"Saved TXT to: {output_file}")
    return str(output_file)

def numpy_to_jpg(input_path, output_root):
    npath = Path(input_path)

    # matches your DICOM naming convention
    output_path = output_root / Path(*npath.parts[-2:]) / "images"

    os.makedirs(output_path, exist_ok=True)

    npy_files = natsorted(npath.glob("*.npy"))

    for img_i, f in enumerate(npy_files):
        img = np.load(f)

        # normalize if needed
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = 255 * (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.astype(np.uint8)

        # grayscale -> 3 channel like your DICOM version
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        filename = output_path / f"{img_i:05d}.jpg"
        cv2.imwrite(str(filename), img_rgb)

    return str(output_path)

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

def compare_2graphs(x0, x1, y0, y1, path, labels=['SAM 2++ Tracking', 'Manual Tracking'], title='Motion in x and y-Direction Over Time', filename='manual_v_template'):
    #create motion graphs to compare
    dt = 6.31*16/1000 # time resolution in second 
    dy = 1.5625 # spatial resolution
    dx = 1.5625 # spatial resolution
    time = np.arange(len(x0)) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    dxx_mm = np.array(x0) * dx
    dyy_mm = np.array(y0) * dy
    dxx_mm_npy = np.array(x1) * dx
    dyy_mm_npy = np.array(y1) * dy

    #already filtered
    #f_dxx = lowpass_filter(np.array(x0) * dx) 
    #f_npy_dxx = lowpass_filter(np.array(x1) * dx)
    #f_dyy = lowpass_filter(np.array(y0) * dy)
    #f_npy_dyy = lowpass_filter(np.array(y1) * dy)

    ax1.plot(time, dxx_mm, label=labels[0])
    ax1.plot(time, dxx_mm_npy, label=labels[1])
    ax1.set_title(title)
    ax1.set_ylabel('x Range (mm)')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, dyy_mm, label=labels[0])
    ax2.plot(time, dyy_mm_npy, label=labels[1])
    ax2.set_ylabel('y Range (mm)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)

    x_lim = ax1.get_ylim()
    y_lim = ax2.get_ylim()
    x_range = abs(x_lim[1]-x_lim[0])
    y_range = abs(y_lim[1]-y_lim[0])
    diff = y_range - x_range
    if (diff > 0):
        ax1.set_ylim(x_lim[0]-(diff/2), x_lim[1]+(diff/2))
    elif (diff < 0):
        ax2.set_ylim(y_lim[0]+(diff/2), y_lim[1]-(diff/2))

    plt.tight_layout()
    output = os.path.join(path, f"{filename}")
    plt.savefig(output, dpi=300)
    #plt.show()

def print_statistics(baseline, prediction, path, folder, runtime=None):
    x0 = baseline[:, 0]
    y0 = baseline[:, 1]
    x1 = prediction[:, 0]
    y1 = prediction[:, 1]

    compare_2graphs(x0, x1, y0, y1, path, title=f'{folder}: Motion in x and y-Direction Over Time',)

    print('Statistics:')

    #mean squared error
    #x0 -= x0[0]
    #x1 -= x1[0]
    #y0 -= y0[0]
    #y1 -= y1[0]
    mse_value_y = (mean_squared_error(y0, y1))**0.5
    mse_value_x = (mean_squared_error(x0, x1))**0.5

    print('x:',mse_value_x)
    print('y:',mse_value_y)

    #correlation coefficient
    x_cc, x_p = pearsonr(x0, x1)
    y_cc, y_p = pearsonr(y0, y1)

    print('x corr coeff & p-val:', x_cc,',', x_p)
    print('y corr coeff & p-val:', y_cc,',', y_p)

    avg_time = runtime / len(x0)

    if (path == None):
        return
    else:
        output = os.path.join(path, f"statistics.txt")
        with open(output, 'w') as file:
            file.write(f"{folder}\n\n")
            file.write(f"Root mean squared error:\n\ty: {mse_value_y}\n\tx: {mse_value_x}\n\n")
            file.write(f"Correlation coefficient:\n\ty corr coeff & p-val: {y_cc} , {y_p}\n\n")
            file.write(f"\tx corr coeff & p-val: {x_cc} , {x_p}\n")
            if runtime:
                file.write(f"Runtime: {runtime} secs\n")
                file.write(f"Runtime per frame: {avg_time} secs")
            file.write(f"\n\n\n{x0}\n{y0}\n\n{x1}\n{y1}")

def load_coords(directory):
    coords_list = []
    for file in natsorted(os.listdir(directory)):
       if file.endswith(".npy"):
            path = os.path.join(directory, file)
            coords = np.load(path)
            coords_list.append(coords)
    return np.stack(coords_list)

def ema_filter_rt(data, state=None, alpha=0.4):
    x, y = data
    if state is None:
        return (x, y), (x, y)

    x_prev, y_prev = state
    x_f = alpha * x + (1 - alpha) * x_prev
    y_f = alpha * y + (1 - alpha) * y_prev
    return (x_f, y_f), (x_f, y_f) 

def multi_np_to_npz(point_tracking_root, input_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    point_folders = [
        d for d in os.listdir(point_tracking_root)
        if os.path.isdir(os.path.join(point_tracking_root, d))
    ]
    point_folders = natsorted(point_folders)

    if len(point_folders) == 0:
        raise ValueError(f"No point folders found in {point_tracking_root}")

    all_points = []
    expected_num_frames = None

    for point_name in point_folders:
        point_dir = os.path.join(point_tracking_root, point_name)

        coord_files = [
            f for f in os.listdir(point_dir)
            if f.endswith(".npy")
        ]
        coord_files = natsorted(coord_files)

        if len(coord_files) == 0:
            raise ValueError(f"No .npy files found in {point_dir}")

        point_coords = []
        for fname in coord_files:
            fpath = os.path.join(point_dir, fname)
            arr = np.load(fpath)
            arr = np.asarray(arr, dtype=np.float32)

            if arr.ndim == 1 and arr.shape[0] == 2:
                pass
            elif arr.ndim == 2 and arr.shape == (1, 2):
                arr = arr[0]
            else:
                raise ValueError(
                    f"File {fpath} has shape {arr.shape}, expected (2,) or (1,2)"
                )

            point_coords.append(arr)

        point_coords = np.stack(point_coords, axis=0)  # (num_frames, 2)

        if expected_num_frames is None:
            expected_num_frames = point_coords.shape[0]
        elif point_coords.shape[0] != expected_num_frames:
            raise ValueError(
                f"Folder {point_dir} has {point_coords.shape[0]} frames, "
                f"expected {expected_num_frames}"
            )

        all_points.append(point_coords)

    trajs_2d = np.stack(all_points, axis=1)  # (num_frames, num_points, 2)
    num_frames, num_points, _ = trajs_2d.shape
    visibs = np.ones((num_frames, num_points), dtype=bool)
    size = (0, 0)

    video_name = os.path.basename(os.path.normpath(input_data))
    out_path = os.path.join(output_dir, f"{video_name}.npz")

    np.savez(out_path, trajs_2d=trajs_2d, visibs=visibs, size=size)

    print(f"Saved NPZ to: {out_path}")
    print(f"Point folders: {point_folders}")
    print(f"trajs_2d shape: {trajs_2d.shape}")
    print(f"visibs shape: {visibs.shape}")

    return out_path, point_folders

def print_multi_stats(prediction, path, num_points, runtime=None):
    dt = 6.31*16/1000 # time resolution in second 
    dy = 1.5625 # spatial resolution
    dx = 1.5625 # spatial resolution

    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    for i in range(num_points):          
        x = prediction[:, i, 0] * dx
        y = prediction[:, i, 1] * dy

        time = np.arange(len(x)) * dt

        axs[0].plot(time, x, label=f'Point {i}')
        axs[1].plot(time, y, label=f'Point {i}')

    axs[0].set_ylabel("X Position (mm)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_title("X Position vs Time")
    axs[0].legend()

    axs[1].set_ylabel("Y Position (mm)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_title("Y Position vs Time")
    axs[1].legend()

    axs[0].grid(True)
    axs[1].grid(True)

    plt.tight_layout()
    output = os.path.join(path, "xy_comparison")
    plt.savefig(output, dpi=300)

    avg_time = runtime / len(prediction[:, 0, 0])

    if (path == None):
        return
    else:
        output = os.path.join(path, f"statistics.txt")
        with open(output, 'w') as file:
            if runtime:
                file.write(f"Runtime: {runtime} secs\n")
                file.write(f"Runtime per frame: {avg_time} secs")

class Kalman2D:
    def __init__(self, dt=1):

        self.initialized = False
        self.x = np.zeros((4,1))

        self.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.P = np.eye(4) * 500
        self.R = np.eye(2) * 10
        self.Q = np.eye(4) * 0.1

    def predict(self):

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[0,0], self.x[1,0]
    
    def update(self, measurement):

        if not self.initialized:
            self.x[0,0] = measurement[0]
            self.x[1,0] = measurement[1]
            self.initialized = True
            return measurement

        z = np.array([[measurement[0]],[measurement[1]]])

        # prediction
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # correction
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[0,0], self.x[1,0]