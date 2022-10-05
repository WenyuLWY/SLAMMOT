cmake_minimum_required(VERSION 3.0.2)
project(object_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  rospy
)




catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES object_detection
    CATKIN_DEPENDS roscpp std_msgs rospy
#  DEPENDS system_lib
)

include_directories(
  include
	${catkin_INCLUDE_DIRS})









