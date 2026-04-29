import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import logging
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from sam2_plus.build_sam import build_sam2_video_predictor_plus

from tools.visualization import show_mask, show_box, show_points
from tools.vos_inference import load_ann_png, get_per_obj_mask, DAVIS_PALETTE, save_masks_to_dir
from tools.sot_inference import save_boxes_to_dir, save_masks_and_boxes_to_dir
from training.dataset_plus.box.utils import np_box_xywh_to_xyxy, np_box_xyxy_to_xywh, np_masks_to_boxes, np_box_clamp_xywh
from benchmarks.sot_benchmark.datasets.utils import load_text

from run_files.img_input import dicom_to_jpg, numpy_to_npz, load_coords, motion_video, print_statistics, ema_filter_rt, compare_2graphs, print_multi_stats, multi_np_to_npz, Kalman2D, numpy_to_txt

#---- INPUT ----------------------------------------------------------------------------

input_data = '../data/realtime/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152'
manual_boundbox = '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152/liver'
output_root = "./my_data_box" 
start = 23

#---------------------------------------------------------------------------------------

dpath = Path(input_data)
output_full = output_root / Path(*dpath.parts[-2:])
output_full.mkdir(parents=True, exist_ok=True)

predictor = build_sam2_video_predictor_plus(
    config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
    ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
    apply_postprocessing=False,
    hydra_overrides_extra=[
        "++model.non_overlap_masks=" + ("false")
    ],
    vos_optimized=False,
    task='box'
)

input_video_dir = dicom_to_jpg(input_data, output_root, start)
input_box_path = numpy_to_txt(manual_boundbox, input_data, output_root)
output_box_dir = output_full / "output"

score_thresh = 0

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=input_video_dir)

    video_name = os.path.basename(input_video_dir)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(input_video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names = natsorted(frame_names)
    height = inference_state["video_height"]
    width = inference_state["video_width"]

    input_frame_idx = 0     # the frame index we interact with
    object_id = 1           # give a unique id to each object we interact with (it can be any integers)

    input_palette = None
    if os.path.isfile(input_box_path):
        input_box_xywh = load_text(str(input_box_path), delimiter=',', dtype=np.float64, backend='numpy').reshape(-1, 4)[0]
    else:
        print(f"Box file {input_box_path} not found. Using default box.")
        input_box_xywh = [316,385,742,488]
    per_obj_input_box_xyxy = {1: np_box_xywh_to_xyxy(np.array(input_box_xywh))}
    object_box_xyxy = per_obj_input_box_xyxy[object_id]

    frame_idx, obj_ids, masks, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=input_frame_idx,
        obj_id=object_id,
        box=object_box_xyxy,
    )

    # run propagation throughout the video and collect the results in a dict
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    video_boxes_xywh = {}  # video_boxes_xyxy contains the per-frame bounding box results
    for out_frame_idx, out_obj_ids, out_mask_logits, output_box_xyxy, out_obj_score_logits in predictor.propagate_in_video(
        inference_state=inference_state,
    ):
        if torch.any(output_box_xyxy[:,:,0] >= output_box_xyxy[:,:,2]) or torch.any(output_box_xyxy[:,:,1] >= output_box_xyxy[:,:,3]):
            logging.warning(f"Invalid box prediction: {output_box_xyxy}")
    
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask
        per_obj_output_box_xywh = {
            out_obj_id: np_box_clamp_xywh(np_box_xyxy_to_xywh(output_box_xyxy[i].cpu().numpy()))
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_boxes_xywh[out_frame_idx] = per_obj_output_box_xywh
    
    # save the tracking results
    save_boxes_to_dir(
        output_bbox_dir=output_box_dir,
        video_name=video_name,
        video_boxes_xywh=video_boxes_xywh,
    )
    
    # visualize the tracking results
    os.makedirs(os.path.join(output_box_dir, video_name), exist_ok=True)
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc="Visualization Results"):
        plt.clf()
        plt.figure()
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_video_dir, frame_names[out_frame_idx] + ".jpg")))
        for out_obj_id, out_box in video_boxes_xywh[out_frame_idx].items():
            box_xywh = out_box[0]
            box_xyxy = np_box_xywh_to_xyxy(np.array(box_xywh))
            show_box(box_xyxy, plt.gca())
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(output_box_dir, video_name, f"{out_frame_idx:05d}_withbox.png"), dpi=250, bbox_inches='tight', pad_inches=0)
        plt.close()


output_stats_dir = output_full / "output"
input_path = os.path.join(output_box_dir, 'images')
motion_video(input_path, output_box_dir, name = None)