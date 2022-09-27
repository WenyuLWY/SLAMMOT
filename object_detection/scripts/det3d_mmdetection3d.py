#! /usr/bin/python
#sdfsafasf sdfasf /home/wenyu/anaconda3/envs/openmmlab/bin/python

#python libs
from email.mime import image
import struct
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

#ip basic
import depth_map_utils

class objectdetection():
    def __init__(self):
        self.subPointCloud=message_filters.Subscriber ("/points_raw",PointCloud2 )
        self.subCameraInfo=message_filters.Subscriber ("/camera_info",CameraInfo )
        self.subImageDepth=message_filters.Subscriber ("/image_depth",Image )
        self.subImageLeft=message_filters.Subscriber ("/image_left",Image )
        self.ts = message_filters.TimeSynchronizer([self.subPointCloud, self.subCameraInfo,self.subImageDepth, self.subImageLeft], 10)
        self.ts.registerCallback(self.objectdetection_callback)

        self.pubDetectionArrayJSK = rospy.Publisher('/detection/objects_jsk', BoundingBoxArray, queue_size=10)
        self.pubDetectionArrayAuto = rospy.Publisher('/detection/objects_auto', DetectedObjectArray, queue_size=10)
        self.pubPointDense = rospy.Publisher('/points_dense', PointCloud2, queue_size=10)
        self.pubDenseDepth = rospy.Publisher('/dense_depth', Image, queue_size=10)

        # config_file = '/home/wenyu/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        # checkpoint_file = '/home/wenyu/mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        # self.model = init_model(config_file, checkpoint_file, device='cuda:0')
        # self.pcd_path="/home/wenyu/data/kitti_odometry/dataset/sequences/00/velodyne/000000.bin"
        # print("successfully load model")

    def objectdetection_callback(self,pcd,camera_info,image_depth,image_left):
        #matrix P projects lidar points to image

        K=np.array([[camera_info.K[0],camera_info.K[1],camera_info.K[2]],
                                [camera_info.K[3],camera_info.K[4],camera_info.K[5]],
                                [camera_info.K[6],camera_info.K[7],camera_info.K[8]]],dtype=float)

        P_velo_to_cam2=np.array([[camera_info.P[0],camera_info.P[1],camera_info.P[2],camera_info.P[3]],
                                                    [camera_info.P[4],camera_info.P[5],camera_info.P[6],camera_info.P[7]],
                                                    [camera_info.P[8],camera_info.P[9],camera_info.P[10],camera_info.P[11]]],dtype=float)     
                                                              
        P_project=np.vstack([K.dot(P_velo_to_cam2),[0,0,0,1]])

        R_inv=np.linalg.inv(P_velo_to_cam2[0:3,0:3])
        P_inv=np.vstack([np.hstack([R_inv,-R_inv.dot(np.array([P_velo_to_cam2[0:3,3]]).transpose())]),[0,0,0,1]])

        #pointcloud2 to mmdetection3d format
        points=np.array(pc2.read_points_list(pcd))

        bridge = CvBridge()
        sparse_depth = bridge.imgmsg_to_cv2(image_depth, '16UC1')
        image=bridge.imgmsg_to_cv2(image_left, 'bgr8')
        projected_depths = np.float32(sparse_depth / 256.0)
        final_depths, process_dict = depth_map_utils.fill_in_multiscale(
            projected_depths, extrapolate=False, blur_type='bilateral',
            show_process=True)
        depth_image = (final_depths * 256).astype(np.uint16)

        i=0
        image_pcd=np.zeros(shape=(4,camera_info.height*camera_info.width))
        color_pcd=np.zeros(shape=(1,camera_info.height*camera_info.width))
        fields=[PointField('x',0,PointField.FLOAT32,1),
                        PointField('y',4,PointField.FLOAT32,1),
                        PointField('z',8,PointField.FLOAT32,1),
                        PointField('rgba',12,PointField.UINT32,1)]

        for width in range(0,camera_info.width):
            for height in range(0,camera_info.height):

                image_pcd[0,i]=final_depths[height,width]*(width-K[0,2])/K[0,0]
                image_pcd[1,i]=final_depths[height,width]*(height-K[1,2])/K[1,1]
                image_pcd[2,i]=final_depths[height,width]
                image_pcd[3,i]=1
                # color_pcd[0,i]=image[height,width,2]*255
                # color_pcd[1,i]=image[height,width,1]*255
                # color_pcd[2,i]=image[height,width,0]*255
                color_pcd[0,i]=struct.unpack('I',struct.pack('BBBB',image[height,width,0],
                    image[height,width,1],image[height,width,2],255))[0]
                i=i+1
        lidar_pcd= P_inv.dot(image_pcd)
        # lidar_pcd/=lidar_pcd[3,:]
        # print(lidar_pcd.transpose())
        lidar_pcd=np.vstack([lidar_pcd[0:3,:],color_pcd])
        self.pubDenseDepth.publish(bridge.cv2_to_imgmsg(depth_image,'16UC1'))
        self.pubPointDense .publish(pc2.create_cloud(pcd.header,fields,lidar_pcd.transpose()))
        # self.pubPointDense .publish(pc2.create_cloud_xyz32(pcd.header,lidar_pcd[0:3,:].transpose()))


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

def main():
    rospy.init_node('det3d_mmdetection3d',anonymous=True)
    objectdetection()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



