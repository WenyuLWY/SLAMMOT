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
from autoware_msgs.msg import DetectedObjectArray,DetectedObject


class objectdetection():
    def __init__(self):
        self.subPointCloud=rospy.Subscriber ("/points_raw",PointCloud2, self.objectdetection_callback, callback_args=None, queue_size=10 )
        self.pubDetectionArrayJSK = rospy.Publisher('/detection/objects_jsk', BoundingBoxArray, queue_size=10)
        self.pubDetectionArrayAuto = rospy.Publisher('/detection/objects_auto', DetectedObjectArray, queue_size=10)
        self.pubPointCloudFV = rospy.Publisher('/points_frontview', PointCloud2, queue_size=10)
        config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        self.pcd_path="/home/wenyu/data/kitti_odometry/dataset/sequences/00/velodyne/000000.bin"
        print("successfully load model")

    def objectdetection_callback(self,pcd):
        points=np.array(pc2.read_points_list(pcd))

        points_class = get_points_type('LIDAR')
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=None)

        result,  data= inference_detector(self.model, points)
        # result,  data= inference_detector(self.model, self.pcd_path)
        boxes=result[0]['boxes_3d'].tensor.cpu().numpy()
        label=result[0]['labels_3d'].numpy()
        score = result[0]['scores_3d'].numpy()

        detectionarray_jsk = BoundingBoxArray()
        detectionarray_auto=DetectedObjectArray()
        for idx, val in enumerate(score):
            if val > 0.3:
                detection_jsk = BoundingBox()     
                detection_jsk.header = pcd.header
                detection_jsk.pose.position.x=boxes[idx,0]
                detection_jsk.pose.position.y=boxes[idx,1]
                detection_jsk.pose.position.z=boxes[idx,2]
                detection_jsk.dimensions.x= boxes[idx,3]
                detection_jsk.dimensions.y= boxes[idx,4]
                detection_jsk.dimensions.z= boxes[idx,5]
                detection_jsk.pose.orientation.x=0
                detection_jsk.pose.orientation.y=0
                detection_jsk.pose.orientation.z=np.sin(boxes[idx,6]/2)
                detection_jsk.pose.orientation.w=np.cos(boxes[idx,6]/2)     
                detection_jsk.value=val #score of detection_jsk,it is optional
                detection_jsk.label=label[idx] #score of detection,it is optional
                detectionarray_jsk.boxes.append(detection_jsk)

            if val >0.3:
                detection_auto= DetectedObject()
                detection_auto.header = pcd.header
                if label[idx] == 0:
                    detection_auto.label='person'
                elif label[idx] == 1:
                    detection_auto.label='bicycle'
                elif label[idx] == 2:
                    detection_auto.label='car'
                detection_auto.score=val
                detection_auto.valid =True
                detection_auto.pose_reliable = True
                detection_auto.pose.position.x=boxes[idx,0]
                detection_auto.pose.position.y=boxes[idx,1]
                detection_auto.pose.position.z=boxes[idx,2]
                detection_auto.dimensions.x= boxes[idx,3]
                detection_auto.dimensions.y= boxes[idx,4]
                detection_auto.dimensions.z= boxes[idx,5]
                detection_auto.pose.orientation.x=0
                detection_auto.pose.orientation.y=0
                detection_auto.pose.orientation.z=np.sin(boxes[idx,6]/2)
                detection_auto.pose.orientation.w=np.cos(boxes[idx,6]/2)   
                detectionarray_auto.objects.append(detection_auto)
            
        detectionarray_jsk.header=pcd.header
        self.pubDetectionArrayJSK.publish(detectionarray_jsk)

        detectionarray_auto.header=pcd.header
        self.pubDetectionArrayAuto.publish(detectionarray_auto)

        # self.pubPointCloudFV .publish(pcd)

def main():
    rospy.init_node('det3d_mmdetection3d',anonymous=True)
    objectdetection()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



