#! /usr/bin/python
#sdfsafasf sdfasf /home/wenyu/anaconda3/envs/openmmlab/bin/python

#python libs
from email.mime import image
import numpy as np
import os

# #mmdetection3d libs
# from mmdet3d.apis import init_model, inference_detector
# from mmdet3d.core.points import get_points_type

# ros libs
import rospy
import cv2
from cv_bridge import CvBridge
import message_filters 

#msgs
from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from sensor_msgs.msg import  PointCloud2,PointField,Image,CameraInfo
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import  BoundingBoxArray,BoundingBox
from autoware_msgs.msg import DetectedObjectArray,DetectedObject


class objectdetection():
    def __init__(self):
        self.subPointCloud=message_filters.Subscriber ("/points_raw",PointCloud2 )
        self.subCameraInfo=message_filters.Subscriber ("/camera_info",CameraInfo )
        
        self.ts = message_filters.TimeSynchronizer([self.subPointCloud, self.subCameraInfo], 10)
        self.ts.registerCallback(self.objectdetection_callback)

        self.pubDetectionArrayJSK = rospy.Publisher('/detection/objects_jsk', BoundingBoxArray, queue_size=10)
        self.pubDetectionArrayAuto = rospy.Publisher('/detection/objects_auto', DetectedObjectArray, queue_size=10)
        self.pubPointCloudFV = rospy.Publisher('/points_frontview', PointCloud2, queue_size=10)
        self.pubSparseDepth = rospy.Publisher('/sparse_depth', Image, queue_size=10)
        self.pubDenseDepth = rospy.Publisher('/dense_depth', Image, queue_size=10)

        # config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        # checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        # self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        # self.pcd_path="/home/wenyu/data/kitti_odometry/dataset/sequences/00/velodyne/000000.bin"
        # print("successfully load model")

    def objectdetection_callback(self,pcd,camera_info):
        #matrix P projects lidar points to image
        P=np.array([[camera_info.P[0],camera_info.P[1],camera_info.P[2],camera_info.P[3]],
                        [camera_info.P[4],camera_info.P[5],camera_info.P[6],camera_info.P[7]],
                        [camera_info.P[8],camera_info.P[8],camera_info.P[10],camera_info.P[11]]],dtype=float)
        #pointcloud2 to mmdetection3d format
        points=np.array(pc2.read_points_list(pcd))
        bridge = CvBridge()
        sparse_depth=project_depths(points[:,0:3].T,P,[camera_info.height, camera_info.width])
        sparse_depth = np.float32(sparse_depth / 256.0)
        print(sparse_depth.shape)
        self.pubSparseDepth.publish(bridge.cv2_to_imgmsg(sparse_depth,'32FC1'))
        # points_class = get_points_type('LIDAR')
        # points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # result, data = inference_detector(self.model, points_mmdet3d)
        # boxes=result[0]['boxes_3d'].tensor.cpu().numpy()
        # label=result[0]['labels_3d'].numpy()
        # score = result[0]['scores_3d'].numpy()
        # detectionarray_jsk = BoundingBoxArray()
        # detectionarray_auto=DetectedObjectArray()
        # for idx, val in enumerate(score):
        #     if val > 0.5:
        #         # Point_xyz_object=np.array([[boxes[idx,0]],[boxes[idx,1]],[boxes[idx,2]],[1]],dtype=float)
        #         # Point_uz_object=P.dot(Point_xyz_object)
        #         # Point_uz_object/=Point_uz_object[2]
        #         # if Point_uz_object[0]<camera_info.width and Point_uz_object[0]>0 and Point_uz_object[1]<camera_info.height and Point_uz_object[1]>0:
        #             detection_jsk = BoundingBox()     
        #             detection_jsk.header = pcd.header
        #             detection_jsk.pose.position.x=boxes[idx,0]
        #             detection_jsk.pose.position.y=boxes[idx,1]
        #             detection_jsk.pose.position.z=boxes[idx,2]
        #             detection_jsk.dimensions.x= boxes[idx,3]
        #             detection_jsk.dimensions.y= boxes[idx,4]
        #             detection_jsk.dimensions.z= boxes[idx,5]
        #             detection_jsk.pose.orientation.x=0
        #             detection_jsk.pose.orientation.y=0
        #             detection_jsk.pose.orientation.z=np.sin(boxes[idx,6]/2)
        #             detection_jsk.pose.orientation.w=np.cos(boxes[idx,6]/2)     
        #             detection_jsk.value=val #score of detection_jsk,it is optional
        #             detection_jsk.label=label[idx] #score of detection,it is optional
        #             detectionarray_jsk.boxes.append(detection_jsk)

        #             detection_auto= DetectedObject()
        #             detection_auto.header = pcd.header
        #             if label[idx] == 0:
        #                 detection_auto.label='person'
        #             elif label[idx] == 1:
        #                 detection_auto.label='bicycle'
        #             elif label[idx] == 2:
        #                 detection_auto.label='car'
        #             detection_auto.score=val
        #             detection_auto.valid =True
        #             detection_auto.pose_reliable = True
        #             detection_auto.pose.position.x=boxes[idx,0]
        #             detection_auto.pose.position.y=boxes[idx,1]
        #             detection_auto.pose.position.z=boxes[idx,2]
        #             detection_auto.dimensions.x= boxes[idx,3]
        #             detection_auto.dimensions.y= boxes[idx,4]
        #             detection_auto.dimensions.z= boxes[idx,5]
        #             detection_auto.pose.orientation.x=0
        #             detection_auto.pose.orientation.y=0
        #             detection_auto.pose.orientation.z=np.sin(boxes[idx,6]/2)
        #             detection_auto.pose.orientation.w=np.cos(boxes[idx,6]/2)   
        #             detectionarray_auto.objects.append(detection_auto)      
        # detectionarray_jsk.header=pcd.header
        # detectionarray_auto.header=pcd.header

        # idx_fv= points[:, 0] >= 0
        # # reflectances = points[idx_fv, 3]
        # points_fv=points[idx_fv,:].transpose()
        # # reflectances = points[:, 3]
        # # points_fv=points.transpose()
        # points_fv[3,:] = 1

        # points_cam=P.dot(points_fv)
        # points_cam=points_cam/points_cam[2,:]
        # points_crop=[]
        # for i in range(points_cam.shape[1]):
        #     if points_cam[0,i] < camera_info.width and points_cam[0,i]>0 and points_cam[1,i]<camera_info.height and points_cam[1,i]>0 and points_cam[2,i]>=0:
        #         points_crop.append(points_fv[0:3,i].transpose())
        # points_crop = np.array(points_crop)

        # self.pubDetectionArrayJSK.publish(detectionarray_jsk)
        # self.pubPointCloudFV .publish(pc2.create_cloud_xyz32(pcd.header,points_crop))
        # self.pubDetectionArrayAuto.publish(detectionarray_auto)

def project_depths(point_cloud, cam_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.
    Args:
        point_cloud: (3, N) Point cloud in cam0
        cam_p: camera projection matrix
        image_shape: image shape [h, w]
        max_depth: optional, max depth for inversion
    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    all_points = point_cloud[:,point_cloud[0,:]>=0].T

    # Save the depth corresponding to each point
    points_in_img = project_pc_to_image(all_points.T, cam_p)
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points = all_points[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    projected_depths[valid_indices] = [
        max(projected_depths[
            points_in_img_int[1, idx], points_in_img_int[0, idx]],
            all_points[idx, 2])
        for idx in range(points_in_img_int.shape[1])]

    projected_depths[valid_indices] = \
        max_depth - projected_depths[valid_indices]

    return projected_depths.astype(np.float32)

def project_pc_to_image(point_cloud, cam_p):
    """Projects a 3D point cloud to 2D points
    Args:
        point_cloud: (3, N) point cloud
        cam_p: camera projection matrix
    Returns:
        pts_2d: (2, N) projected coordinates [u, v] of the 3D points
    """

    pc_padded = np.append(point_cloud, np.ones((1, point_cloud.shape[1])), axis=0)
    pts_2d = np.dot(cam_p, pc_padded)

    pts_2d[0:2] = pts_2d[0:2] / pts_2d[2]
    return pts_2d[0:2]

def main():
    rospy.init_node('det3d_mmdetection3d',anonymous=True)
    objectdetection()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



