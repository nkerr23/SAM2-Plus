from run_files.img_input import motion_video
import os
from natsort import natsorted

def make_vid():
    output_point_dir = "/Users/nina/Documents/ucla/research/mrrl/w26/SAM2-Plus/demo/data/images"
    input_path = os.path.join(output_point_dir, 'images')
    motion_video(input_path, output_point_dir, name = None)



'''

import numpy as np

data = np.load("./my_data/0119_RTimages/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0154/output/liver_edge/images.npz")["trajs_2d"]

print(data)

folder = '/Users/nina/Documents/ucla/research/mrrl/data/manual_tracking_coords/CV_GARADIAL_GRE3_OBL_SAG_3SL_DYNAMIC_0154/liver_edge'

files = natsorted([
    f for f in os.listdir(folder)
    if f.endswith(".npy")
])

for f in files:
    path = os.path.join(folder, f)
    arr = np.load(path)


    print(arr)
    '''