#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import signal
import os
import time
# from importlib.metadata import files

import rospy
import message_filters 

from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
from sensor_msgs.msg import  PointCloud2,PointField,CameraInfo

import sensor_msgs.point_cloud2 as pc2
from autoware_msgs.msg import DetectedObjectArray,DetectedObject
from jsk_recognition_msgs.msg import  BoundingBoxArray,BoundingBox,TrackingStatus
# from autoware_msgs.msg import Detection3DArray

path = "/home/wenyu/detection/04/" #文件夹目录
# path = "/home/bingda/catkin_ws2/datatest/1/" #文件夹目录

files= os.listdir(path) #得到文件夹下的所有文件名称
files.sort(key=lambda x:int(x[:-4]))

class objectdetection():
    global label_,x,y,z,box
    def __init__(self):
        self.subPointCloud=rospy.Subscriber ("/points_raw",PointCloud2, self.objectdetectionHandler, callback_args=None, queue_size=10 )
        self.pubDetectionArray = rospy.Publisher('/detections/lidar_detector/objects', BoundingBoxArray, queue_size=10)

        # self.subPointCloud=message_filters.Subscriber ("/points_raw",PointCloud2 )
        # self.subCameraInfo=message_filters.Subscriber ("/camera_info",CameraInfo )


        self.pubDetectionArrayJSK = rospy.Publisher('/detection/objects_jsk', BoundingBoxArray, queue_size=10)
        self.pubDetectionArrayAuto = rospy.Publisher('/detection/objects_auto', DetectedObjectArray, queue_size=10)
        # self.pub_tracking = rospy.Publisher('/detection/tracking', TrackingStatus, queue_size=10)
        # self.pubPointCloudFV = rospy.Publisher('/points_frontview', PointCloud2, queue_size=10)
        

        #根据时间戳发布检测结果
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.subPointCloud, self.subCameraInfo], 10, 0.1, allow_headerless=True)
        # self.ts.registerCallback(self.objectdetectionHandler)
        
        # for file in files: #遍历文件夹
        #     a = np.loadtxt(path+file,comments='#')
        #     #两个集合合并
        #     a1 = np.concatenate((a[(a[:, 0] == 0) & (a[:, -1] > 0.3), :], a[(a[:, 0] == 1) & (a[:, -1] > 0.35), :], a[(a[:, 0] == 2) & (a[:, -1] > 0.5), :]), axis=0)
        #     #筛选出最后以列大于0.5的数据
        #     # a1 = a[a[:, -1] > 0.5, :]
        #     a1=(np.around(a1[:,:],2))

        #     # a1=(np.around(a[(a[:, -1] > 0.5), :],2))

        # # a1=firstTxtArr
        # print(a1)
        # self.ts = message_filters.TimeSynchronizer([self.subPointCloud, a1], 10)
        # self.ts.registerCallback(self.objectdetectionHandler)
    
        
        # for file in files: #遍历文件夹
        #     a = np.loadtxt(path+file,comments='#')
        #     #两个集合合并
        #     a1 = np.concatenate((a[(a[:, 0] == 0) & (a[:, -1] > 0.3), :], a[(a[:, 0] == 1) & (a[:, -1] > 0.35), :], a[(a[:, 0] == 2) & (a[:, -1] > 0.5), :]), axis=0)
        #     #筛选出最后以列大于0.5的数据
        #     # a1 = a[a[:, -1] > 0.5, :]
        #     a1=(np.around(a1[:,:],2))
        #     print(a1)

       
        # laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        # laser_cloud_msg.header.frame_id = "/velodyne";
        # pub_laser_cloud.publish(laser_cloud_msg);

    
    # def callback(self,pcd):
    def objectdetectionHandler(self,pcd):
        points=np.array(pc2.read_points_list(pcd))
        detectionarray_jsk = BoundingBoxArray()
        detectionarray_auto=DetectedObjectArray()
        #读取文件的目录
        #当文件夹下的文件未读完时，继续读取
        # while(x<files.size):
        for file in files: #遍历文件夹
            a = np.loadtxt(path+file,comments='#')
            #两个集合合并
            a1 = np.concatenate((a[(a[:, 0] == 0) & (a[:, -1] > 0.3), :], a[(a[:, 0] == 1) & (a[:, -1] > 0.35), :], a[(a[:, 0] == 2) & (a[:, -1] > 0.5), :]), axis=0)
            #筛选出最后以列大于0.5的数据
            # a1 = a[a[:, -1] > 0.5, :]
            a1=(np.around(a1[:,:],2))
            #让index在a1的所有行中循环
            for idx in range(a1.shape[0]):
                    #打印idx,换行
                    # # print(idx)
                    # print(a1)
                    #输出a1的行数
                    print("a1是",a1)
                    print("行数是",a1.shape[0])
                              
                    Point_xyz_object=np.array([[a1[idx,1]],[a1[idx,2]],[a1[idx,3]]],dtype=float)
                    detection_jsk = BoundingBox()     
                    detection_jsk.header = pcd.header
                    detection_jsk.header.stamp = pcd.header.stamp
                    #进行跟踪
                    # if(a1[idx,0]==0):
                    detection_jsk.label = 1
                    detection_jsk.pose.position.x = Point_xyz_object[0]
                    detection_jsk.pose.position.y = Point_xyz_object[1]
                    detection_jsk.pose.position.z = Point_xyz_object[2]
                    detection_jsk.dimensions.x = a1[idx,4]
                    detection_jsk.dimensions.y = a1[idx,5]
                    detection_jsk.dimensions.z = a1[idx,6]
                    detection_jsk.value = a1[idx,8]
                    detection_jsk.label=a1[idx,0]
                    detectionarray_jsk.boxes.append(detection_jsk)
                    istracking=False
                    detectionarray_jsk.header=pcd.header
                    detectionarray_jsk.header.stamp=pcd.header.stamp
                    self.pubDetectionArrayJSK.publish(detectionarray_jsk)
                    #清空detectionarray_jsk
                    # detectionarray_jsk.boxes.clear()

                        # self.pub_tracking.publish(istracking)
                    # istracking.status=TrackingStatus.NOT_TRACKED
                    # # detection_jsk.pose.position.x 等于 a1第idx行第1列的数据
                    # detection_jsk.pose.position.x=a1[idx,1]
                    # detection_jsk.pose.position.y=a1[idx,2]
                    # detection_jsk.pose.position.z=a1[idx,3]
                    # detection_jsk.dimensions.x= a1[idx,4]
                    # detection_jsk.dimensions.y= a1[idx,5]
                    # detection_jsk.dimensions.z= a1[idx,6]
                    # detection_jsk.pose.orientation.x=0
                    # detection_jsk.pose.orientation.y=0
                    # detection_jsk.pose.orientation.z=np.sin(a1[idx,7]/2)
                    # detection_jsk.pose.orientation.w=np.cos(a1[idx,7]/2)     
                    # detection_jsk.value=a1[idx,8] #score of detection_jsk,it is optional
                    # detection_jsk.label=a1[idx,0] #score of detection,it is optional
                    # detectionarray_jsk.boxes.append(detection_jsk)
            


                    detection_auto= DetectedObject()
                    detection_auto.header = pcd.header
                    if a1[idx,0]== 0:
                        detection_auto.label='person'
                    elif a1[idx,0] == 1:
                        detection_auto.label='bicycle'
                    elif a1[idx,0] == 2:
                        detection_auto.label='car'
                    # detection_auto.score=val
                    detection_auto.valid =True
                    detection_auto.pose_reliable = True
                    detection_auto.pose.position.x=a1[idx,1]
                    detection_auto.pose.position.y=a1[idx,2]
                    detection_auto.pose.position.z=a1[idx,3]
                    detection_auto.dimensions.x= a1[idx,4]
                    detection_auto.dimensions.y= a1[idx,5]
                    detection_auto.dimensions.z= a1[idx,6]
                    detection_auto.pose.orientation.x=0
                    detection_auto.pose.orientation.y=0
                    detection_auto.pose.orientation.z=np.sin(a1[idx,7]/2)
                    detection_auto.pose.orientation.w=np.cos(a1[idx,7]/2)   
                    detectionarray_auto.objects.append(detection_auto)
                    
            #将a1中原有的数据清空
            a1 = np.empty((0, 9), float)
            print("a1清空后:")
            print(a1)
            #输出BoundingBoxArray的个数
            # print(len(detectionarray_jsk.boxes))
            # print("",a1.shape[0])

            



        

            detectionarray_jsk.header=pcd.header
            detectionarray_jsk.header.stamp=pcd.header.stamp





            detectionarray_auto.header=pcd.header
            detectionarray_auto.header.stamp=pcd.header.stamp

            
            # self.pubDetectionArrayJSK.publish(detectionarray_jsk)
            # self.pubIsTracking.publish(istracking)
            # self.pubPointCloudFV .publish(pc2.create_cloud_xyz32(pcd.header,points_crop))
            self.pubDetectionArrayAuto.publish(detectionarray_auto)
            time.sleep(1)



def main():
    rospy.init_node('det3d_load_file',anonymous=True)
    objectdetection()
    rospy.spin()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down")



