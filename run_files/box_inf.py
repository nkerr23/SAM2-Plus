import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tempfile
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

from run_files.img_input import dicom_to_jpg, numpy_to_npz, numpy_to_jpg, load_coords, motion_video, print_statistics, ema_filter_rt, compare_2graphs, print_multi_stats, multi_np_to_npz, Kalman2D, numpy_to_txt

#---- INPUT ----------------------------------------------------------------------------
'''
input_data = '../data/realtime_npy/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152'
manual_boundbox = '../data/bounding_boxes/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0152/liver'
output_root = "./my_data_box" 
'''
#---- Hoffman Cluster GPU COMMAND: qrsh -l gpu,A6000,cuda=1 -------------------------------
'''
dpath = Path(input_data)
output_full = output_root / Path(*dpath.parts[-2:])
output_full.mkdir(parents=True, exist_ok=True)

input_video_dir = numpy_to_jpg(input_data, output_root) #replaced by tmp dir
input_box_path = numpy_to_txt(manual_boundbox, input_data, output_root)
output_box_dir = output_full / "output" # set input data to array input frames
'''
#---- build SAM2++ predictor -----------------------------------------------------------------

def box_predictor() -> object:
    return build_sam2_video_predictor_plus(
        config_file="configs/sam2.1/sam2.1_hiera_b+_predmasks_decoupled_MAME.yaml",
        ckpt_path="./checkpoints/SAM2-Plus/checkpoint_phase123.pt",
        apply_postprocessing=False,
        hydra_overrides_extra=[
            "++model.non_overlap_masks=false",
            "++model.image_size=896",
            "++model.memory_attention.layer.self_attention.feat_sizes=[56,56]",
            "++model.memory_attention.layer.cross_attention.feat_sizes=[56,56]",
        ],
        vos_optimized=False,
        task='box'
    )


def clamp_box_xywh_to_frame(box_xywh, frame_shape):
    box = np.asarray(box_xywh, dtype=np.float32)
    original_shape = box.shape
    x, y, w, h = box.reshape(4)
    frame_h, frame_w = frame_shape[:2]
    x0 = np.clip(x, 0, frame_w)
    y0 = np.clip(y, 0, frame_h)
    x1 = np.clip(x + w, 0, frame_w)
    y1 = np.clip(y + h, 0, frame_h)
    return np.array(
        [x0, y0, max(0, x1 - x0), max(0, y1 - y0)], dtype=np.float32
    ).reshape(original_shape)

#---- main tracking function ------------------------------------------------------------------

def track_boundingbox(
    input_frames: list[np.ndarray],
    input_box_xywh, #(x, y, w, h)
    predictor,
    score_thresh: float = 0,
    box_size_anchor=None,
    max_size_change: float = 0.1,
) -> dict[int, np.ndarray]:

    # input box for first frame
    input_box_xywh = np.asarray(input_box_xywh, dtype=np.float64)
    if box_size_anchor is None:
        box_size_anchor = input_box_xywh[-2:]
    box_size_anchor = np.asarray(box_size_anchor, dtype=np.float64).reshape(2)
    if np.any(box_size_anchor <= 0):
        raise ValueError(f"Box size anchor must be positive, got {box_size_anchor}")
    if not 0 <= max_size_change < 1:
        raise ValueError("max_size_change must be in [0, 1)")

    # write input frames to temporary directory for sam2++ to read from
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, frame in enumerate(input_frames):
            img = Image.fromarray(frame)
            img.save(os.path.join(tmp_dir, f"{idx:05d}.jpg"), quality=95)

        # run model 
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = predictor.init_state(video_path=tmp_dir)

            input_frame_idx = 0     # the frame index we interact with
            object_id = 1           # give a unique id to each object we interact with (it can be any integers)

            object_box_xyxy = np_box_xywh_to_xyxy(input_box_xywh)
            
            frame_idx, obj_ids, masks, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                box=object_box_xyxy,
            )

            # run propagation throughout the video and collect the results in a dict
            video_boxes_xywh = {}  # video_boxes_xyxy contains the per-frame bounding box results

            kf_centers = {}
            filtered_sizes = {}

            for out_frame_idx, out_obj_ids, _, output_box_xyxy, _ in predictor.propagate_in_video(
                inference_state=inference_state,
            ):
                if torch.any(output_box_xyxy[:,:,0] >= output_box_xyxy[:,:,2]) or torch.any(output_box_xyxy[:,:,1] >= output_box_xyxy[:,:,3]):
                    logging.warning(f"Invalid box prediction: {output_box_xyxy}")
            
                '''
                per_obj_output_box_xywh = {
                    out_obj_id: np_box_clamp_xywh(np_box_xyxy_to_xywh(output_box_xyxy[i].cpu().numpy()))
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                '''

                per_obj_output_box_xywh = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    raw_xywh = np_box_clamp_xywh(np_box_xyxy_to_xywh(output_box_xyxy[i].cpu().numpy()))
                    x, y, w, h = raw_xywh.squeeze()

                    if out_obj_id not in kf_centers:
                        kf_centers[out_obj_id] = Kalman2D()
                        filtered_sizes[out_obj_id] = box_size_anchor.copy()

                    # filter center position
                    cx_filt, cy_filt = kf_centers[out_obj_id].update((x + w / 2, y + h / 2))

                    # Size has no meaningful velocity. Smooth it and keep it near the
                    # original prompt so model errors cannot compound across windows.
                    min_size = box_size_anchor * (1 - max_size_change)
                    max_size = box_size_anchor * (1 + max_size_change)
                    measured_size = np.clip((w, h), min_size, max_size)
                    filtered_sizes[out_obj_id] = (
                        0.8 * filtered_sizes[out_obj_id] + 0.2 * measured_size
                    )
                    w_filt, h_filt = filtered_sizes[out_obj_id]

                    # reconstruct xywh from filtered center and size
                    filtered_xywh = np.array([
                        cx_filt - w_filt / 2,
                        cy_filt - h_filt / 2,
                        w_filt,
                        h_filt,
                    ], dtype=np.float32).reshape(raw_xywh.shape)

                    per_obj_output_box_xywh[out_obj_id] = clamp_box_xywh_to_frame(
                        filtered_xywh, input_frames[out_frame_idx].shape
                    )

                video_boxes_xywh[out_frame_idx] = per_obj_output_box_xywh

    return video_boxes_xywh
            
'''
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
'''
'''
output_stats_dir = output_full / "output"
input_path = os.path.join(output_box_dir, 'images')
motion_video(input_path, output_box_dir, name = None)
'''
