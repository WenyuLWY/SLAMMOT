import glob
import os
import sys
import time

import cv2
import numpy as np
import png

from ip_basic_m.ip_basic import depth_map_utils
from ip_basic_m.ip_basic import vis_utils


# pcd to sparse depth ,generate dense depth, dense depth to pcd
# sparse depth map
input_depth_dir = os.path.expanduser(
    '~/data/kitti_depth_completion/data/depth_selection/val_selection_cropped/velodyne_raw')
data_split = 'val'

fill_type = 'multiscale'
extrapolate = False
blur_type = 'bilateral'
save_output = False
show_process = True
save_depth_maps = False

# fill_type = 'fast'
# extrapolate = True
# blur_type = 'gaussian'
# save_output = True
# show_process = False
# save_depth_maps = True

# fill_type = 'fast'
# extrapolate = False
# blur_type = 'bilateral'
# save_output = True
# show_process = False
# save_depth_maps = True

depth_image_path="/home/wenyu/data/kitti_depth_completion/data/depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png"

depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)


projected_depths = np.float32(depth_image / 256.0)

if fill_type == 'fast':
    final_depths = depth_map_utils.fill_in_fast(
        projected_depths, extrapolate=extrapolate, blur_type=blur_type)
elif fill_type == 'multiscale':
    final_depths, process_dict = depth_map_utils.fill_in_multiscale(
        projected_depths, extrapolate=extrapolate, blur_type=blur_type,
        show_process=show_process)
else:
    raise ValueError('Invalid fill_type {}'.format(fill_type))

depth_image = (final_depths * 256).astype(np.uint16)
cv2.imshow("depth_image",depth_image)
cv2.waitKey()
