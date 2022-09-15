#! /home/wenyu/anaconda3/envs/openmmlab/bin/python
from logging import shutdown
import numpy as np
import sys
import signal
import os

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.points import get_points_type
import cv2
from cv_bridge import CvBridge
import rospy

from sensor_msgs.msg import  PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import  BoundingBoxArray,BoundingBox
# from autoware_msgs.msg import Detection3DArray





class objectdetection():
    def __init__(self):
        self.subPointCloud=rospy.Subscriber ("/kitti/velo/pointcloud",PointCloud2, self.objectdetectionHandler, callback_args=None, queue_size=10 )
        self.pubDetectionArray = rospy.Publisher('/detections/lidar_detector/objects', BoundingBoxArray, queue_size=10)
        config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        print("successfully load model")

    def objectdetectionHandler(self,pcd):
        points=np.array(pc2.read_points_list(pcd))

        points_class = get_points_type('LIDAR')
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)

        result,  data= inference_detector(self.model, points)
        # result, data = inference_detector(self.model, self.pcd_path)
        boxes=result[0]['boxes_3d'].tensor.cpu().numpy()
        label=result[0]['labels_3d']
        score = result[0]['scores_3d']
        detectionarray = BoundingBoxArray()
        for rows, cols in enumerate(boxes):
            detection = BoundingBox()
            detection.header = pcd.header
            # detection.pose.orientation.w = 1
            detection.pose.position.x=boxes[rows,0]
            detection.pose.position.y=boxes[rows,1]
            detection.pose.position.z=boxes[rows,2]
            detection.dimensions.x= boxes[rows,3]
            detection.dimensions.y= boxes[rows,4]
            detection.dimensions.z= boxes[rows,5]
            detectionarray.boxes.append(detection) 
        detectionarray.header=pcd.header
        self.pubDetectionArray.publish(detectionarray)


def main():
    rospy.init_node('det3d_load_file',anonymous=True)
    objectdetection()
    rospy.spin()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



