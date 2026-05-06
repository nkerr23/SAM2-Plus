import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from sam2_plus.build_sam import build_sam2_video_predictor_plus
from tools.visualization import show_mask, show_box, show_points
from tools.vos_inference import load_ann_png, get_per_obj_mask, DAVIS_PALETTE, save_masks_to_dir

from run_files.img_input import dicom_to_jpg, numpy_to_npz, load_coords, motion_video, print_statistics, ema_filter_rt, compare_2graphs, print_multi_stats, multi_np_to_npz, Kalman2D, numpy_to_jpg

#---- INPUT ----------------------------------------------------------------------------

input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0347'
seg_mask = '../data/0119_0347.png'
output_root = "./my_data_kal" 

#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------

dpath = Path(input_data)
output_full = output_root / Path(*dpath.parts[-2:])
output_full.mkdir(parents=True, exist_ok=True)

input_video_dir = numpy_to_jpg(input_data, output_root)
output_mask_dir = output_full / "output"
overlay_output_dir = output_mask_dir / "overlay"

os.makedirs(overlay_output_dir, exist_ok=True)

start_time = datetime.now()

predictor = build_sam2_video_predictor_plus(
    config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
    ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
    apply_postprocessing=False,
    hydra_overrides_extra=[
        "++model.non_overlap_masks=" + ("false")
    ],
    vos_optimized=False,
    task='mask'
)

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
    object_id = 255           # give a unique id to each object we interact with (it can be any integers)

    input_palette = None
    input_mask, input_palette = load_ann_png(seg_mask)
    per_obj_input_mask = get_per_obj_mask(input_mask)
    object_mask = per_obj_input_mask[object_id]

    predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=input_frame_idx,
        obj_id=object_id,
        mask=object_mask,
    )

    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits, _, _ in predictor.propagate_in_video(
        inference_state
    ):
        per_obj_output_mask = {
            out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        video_segments[out_frame_idx] = per_obj_output_mask

    end_time = datetime.now()
    runtime = end_time-start_time
    print(runtime)
    
    # write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=False,
            output_palette=output_palette,
        )

    os.makedirs(os.path.join(overlay_output_dir, video_name), exist_ok=True)
    
    # visualize the tracking results
    for out_frame_idx in tqdm(range(0, len(frame_names)), desc="Visualization Results"):
        plt.clf()
        plt.figure()
        # plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(input_video_dir, frame_names[out_frame_idx] + ".jpg")))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{overlay_output_dir}/{out_frame_idx:05d}_withMask.png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    #save video
    input_path = os.path.join(overlay_output_dir)
    motion_video(input_path, overlay_output_dir, name = None)