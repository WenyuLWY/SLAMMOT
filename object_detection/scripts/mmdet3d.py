#! /home/wenyu/anaconda3/envs/openmmlab/bin/python

from lib2to3.pgen2.token import RPAR
from logging import shutdown
import rospy
from mmdet3d.apis import init_model, inference_detector
import numpy as np
import os

config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

pcd="/home/wenyu/data/kitti_odometry/dataset/sequences/00/velodyne/000000.bin"

# save=np.vstack([label,boxes.T,score]).T

if __name__ == '__main__':
    rospy.init_node('det3d',anonymous=True)
    rate=rospy.Rate(10)
    while not rospy is shutdown:
        rospy.loginfo("hello")
        result, data = inference_detector(model, pcd)
        boxes=result[0]['boxes_3d'].tensor.cpu().numpy()
        label=result[0]['labels_3d']
        score = result[0]['scores_3d']
        print(boxes)
