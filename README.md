# SLAM

roslaunch aloam_velodyne kitti_helper.launch
 
roslaunch aloam_velodyne aloam_velodyne_HDL_64.launch

rosbag play /home/wenyu/data/kitti_odometry/dataset/rosbag/kitti_00.bag /kitti/velo/pointcloud:=/velodyne_points

evo_ape /home/wenyu/data/kitti_odometry/dataset/results/00.txt kitti_odo_00.txt -va --plot --plot_mode xz 

evo_traj kitti kitti_odo_00.txt --ref=/home/wenyu/data/kitti_odometry/dataset/results/00.txt -p --plot_mode=xz

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS Melodic**
### 1.2 **Msgs** 
sudo apt install ros-melodic-vision-msgs

sudo apt-get install ros-melodic-jsk-recognition-msgs

sudo apt-get install ros-melodic-jsk-rviz-plugins

+ ceres

sudo apt install libceres-dev

+ gtsam

sudo add-apt-repository ppa:borglab/gtsam-release-4.0

sudo apt install libgtsam-dev libgtsam-unstable-dev
