#! /home/wenyu/anaconda3/envs/openmmlab/bin/python

#python libs
import numpy as np
import os

#mmdetection3d libs
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.points import get_points_type

# ros libs
import rospy
import cv2
from cv_bridge import CvBridge

#msgs
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from sensor_msgs.msg import  PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import  BoundingBoxArray,BoundingBox
# from autoware_msgs.msg import Detection3DArray

class objectdetection():
    def __init__(self):
        self.subPointCloud=rospy.Subscriber ("/points_raw",PointCloud2, self.objectdetectionHandler, callback_args=None, queue_size=10 )
        self.pubDetectionArray = rospy.Publisher('/detections/lidar_detector/objects', BoundingBoxArray, queue_size=10)
        self.pubPointCloudFV = rospy.Publisher('/points_frontview', PointCloud2, queue_size=10)
        config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        self.pcd_path="/home/wenyu/data/kitti_odometry/dataset/sequences/00/velodyne/000000.bin"
        print("successfully load model")

    def objectdetectionHandler(self,pcd):
        points=np.array(pc2.read_points_list(pcd))

        points_class = get_points_type('LIDAR')
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)

        result,  data= inference_detector(self.model, points)
        # result,  data= inference_detector(self.model, self.pcd_path)
        boxes=result[0]['boxes_3d'].tensor.cpu().numpy()
        label=result[0]['labels_3d'].numpy()
        score = result[0]['scores_3d'].numpy()

        detectionarray = BoundingBoxArray()
        for idx, val in enumerate(score):
            if val > 0.3:
                detection = BoundingBox()     
                detection.header = pcd.header
                detection.pose.position.x=boxes[idx,0]
                detection.pose.position.y=boxes[idx,1]
                detection.pose.position.z=boxes[idx,2]
                detection.dimensions.x= boxes[idx,3]
                detection.dimensions.y= boxes[idx,4]
                detection.dimensions.z= boxes[idx,5]
                detection.pose.orientation.x=0
                detection.pose.orientation.y=0
                detection.pose.orientation.z=np.sin(boxes[idx,6]/2)
                detection.pose.orientation.w=np.cos(boxes[idx,6]/2)     
                detection.value=val #score of detection,it is optional
                detection.label=label[idx] #score of detection,it is optional
                detectionarray.boxes.append(detection) 
        detectionarray.header=pcd.header
        self.pubDetectionArray.publish(detectionarray)
        self.pubPointCloudFV .publish(pcd)

def main():
    rospy.init_node('det3d_mmdetection3d',anonymous=True)
    objectdetection()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



