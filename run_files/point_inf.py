import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from natsort import natsorted
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from sam2_plus.build_sam import build_sam2_video_predictor_plus

from tools.visualization import show_mask, show_box, show_points
from tools.vos_inference import load_ann_png, get_per_obj_mask, DAVIS_PALETTE, save_masks_to_dir
from tools.pt_inference_plus import load_visible_points_from_npz

from run_files.img_input import dicom_to_jpg, numpy_to_jpg, numpy_to_npz, load_coords, motion_video, print_statistics, ema_filter_rt, compare_2graphs, print_multi_stats, multi_np_to_npz, Kalman2D

#---- INPUT ----------------------------------------------------------------------------

input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0347'
manual_points = '../data/manual_tracking_coords/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0347_needleTip'
output_root = "./my_data_kal" 

#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------

dpath = Path(input_data)
output_full = output_root / Path(*dpath.parts[-2:])
output_full.mkdir(parents=True, exist_ok=True)

input_video_dir = numpy_to_jpg(input_data, output_root)
input_point_path, pt_folders = multi_np_to_npz(manual_points, input_data, output_root)
output_point_dir = output_full / "output"

predictor = build_sam2_video_predictor_plus(
    config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
    ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
    apply_postprocessing=False,
    hydra_overrides_extra=[
        "++model.non_overlap_masks=" + ("false")
    ],
    vos_optimized=False,
    task='point'
)

radius, sigma = 5, 2
score_thresh = 0

start_time = datetime.now()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    video_name = os.path.basename(input_video_dir)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(input_video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names = natsorted(frame_names)

    inference_state = predictor.init_state(video_path=input_video_dir)
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    input_frame_idx = 0     # the frame index we interact with
    #object_id = 0           # give a unique id to each object we interact with (it can be any integers)
    #num_frames, num_points = len(frame_names), 1

    input_data = np.load(input_point_path, allow_pickle=True)
    input_point, input_visible = torch.tensor(input_data['trajs_2d'].astype(np.float32)), torch.tensor(input_data['visibs'].astype(bool))
    per_obj_input_point = load_visible_points_from_npz(
        input_points=input_point,
        input_visibles=input_visible,
        frame_idx=input_frame_idx,
    )

    num_frames, num_points = input_point.shape[0], input_point.shape[1]

    for object_id in range(num_points):
        object_point = per_obj_input_point[object_id]

        predictor.add_new_points_and_generate_gaussian_mask(
            inference_state=inference_state,
            frame_idx=input_frame_idx,
            obj_id=object_id,
            points=object_point.unsqueeze(0).numpy(),
            labels=np.array([1]),
            radius=radius,
            sigma=sigma,
        )

    data = np.load(input_point_path, allow_pickle=True)
    #print("trajs_2d:", data["trajs_2d"].shape)
    #print("visibs:", data["visibs"].shape)
    #print("frame 0 points:", data["trajs_2d"][:, 0, :])
    #print("frame 0 visible:", data["visibs"][:, 0])

    # run propagation throughout the video and collect the results in a dict
    point_array = -np.ones((num_frames, num_points, 2), dtype=np.float32)
    point_array_unfilt = point_array.copy()
    visible_array = np.zeros((num_frames, num_points), dtype=bool)

    #filter_states = {i: None for i in range(num_points)}
    kalman_filters = {i: Kalman2D() for i in range(num_points)}
    alpha = 0.5

    for out_frame_idx, out_obj_ids, out_mask_logits, out_box_xyxys, out_obj_score_logits in predictor.propagate_in_video(
        inference_state
    ):
        for out_obj_id, out_mask_logit, out_obj_score_logit in zip(out_obj_ids, out_mask_logits, out_obj_score_logits):
            out_mask_logit, out_obj_score_logit = out_mask_logit.squeeze(0), out_obj_score_logit.squeeze(0)
            max_index = torch.argmax(out_mask_logit)
            max_score_y, max_score_x = torch.unravel_index(max_index, out_mask_logit.shape)

            point_array_unfilt[out_frame_idx, out_obj_id] = np.array([max_score_x.item(), max_score_y.item()], dtype=np.float32)
            
            #raw_point = (float(max_score_x.item()), float(max_score_y.item()))
            ##filtered_point, filter_state = ema_filter_rt(raw_point, filter_state, alpha)
            #filtered_point, filter_states[out_obj_id] = ema_filter_rt(raw_point,filter_states[out_obj_id],alpha)

            raw_point = (float(max_score_x.item()), float(max_score_y.item()))
            filtered_x, filtered_y = kalman_filters[out_obj_id].update(raw_point)
            filtered_point = (filtered_x, filtered_y)

            point_array[out_frame_idx, out_obj_id] = np.array(filtered_point, dtype=np.float32)
            
            visible_array[out_frame_idx, out_obj_id] = (out_obj_score_logit > score_thresh).cpu().numpy()

    end_time = datetime.now()
    runtime = end_time-start_time
    print(runtime)
    
    # write the output masks as palette PNG files to output_mask_dir
    os.makedirs(output_point_dir, exist_ok=True)
    np.savez(os.path.join(output_point_dir, f"{video_name}.npz"), trajs_2d=point_array, visibs=visible_array, size=(width, height))

    # visualize the tracking results
    os.makedirs(os.path.join(output_point_dir, video_name), exist_ok=True)
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc="Visualization Results"):
        plt.clf()
        plt.figure()
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_video_dir, frame_names[out_frame_idx] + ".jpg")))

        #points = point_array[out_frame_idx, object_id].reshape(1, 2)
        #labels = np.array([-1], np.int32)
        #show_points(points, labels, plt.gca(), marker_size=20, edgecolor=None)
        points = point_array[out_frame_idx]
        labels = np.ones(num_points, dtype=np.int32)

        #print("frame", out_frame_idx, "points:", points)
        show_points(points, labels, plt.gca(), marker_size=20)

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(output_point_dir, video_name, f"{out_frame_idx:05d}_withPoint.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    #save video
    input_path = os.path.join(output_point_dir, 'images')
    motion_video(input_path, output_point_dir, name = None)

    print_multi_stats(point_array, output_point_dir, num_points, runtime)

    gt_array = np.load(input_point_path, allow_pickle=True)['trajs_2d']  # (num_frames, num_points, 2)

    for point_idx, folder in enumerate(pt_folders):
        single_output = os.path.join(output_point_dir, folder)
        os.makedirs(single_output, exist_ok=True)

        gt = gt_array[:, point_idx, :]         # (num_frames, 2)
        pred = point_array[:, point_idx, :]    # (num_frames, 2)

        x0 = point_array_unfilt[:, point_idx, 0]
        y0 = point_array_unfilt[:, point_idx, 1]
        x1 = point_array[:, point_idx, 0]
        y1 = point_array[:, point_idx, 1]

        compare_2graphs(
            x0, x1, y0, y1,
            single_output,
            labels=['Unfiltered', 'Filtered'],
            title=f'{folder}: Motion in x and y-Direction Over Time',
            filename='filtered_v_unfiltered'
        )

        print(f"\nPoint {point_idx} ({folder})")
        print_statistics(gt, pred, single_output, folder, runtime)

    '''
    #compare filtered vs. unfiltered
    x0 = point_array_unfilt[:, 0, 0]
    y0 = point_array_unfilt[:, 0, 1]
    x1 = point_array[:, 0, 0]
    y1 = point_array[:, 0, 1]
    compare_2graphs(x0, x1, y0, y1, output_point_dir, labels=['Unfiltered', 'Filtered'], title='Motion in x and y-Direction Over Time', filename='filtered_v_unfiltered')

    #print stats
    input_array = load_coords(point_tracking)
    output_array = point_array[:,0,:]
    print_statistics(input_array, output_array, output_point_dir, runtime)

    '''



