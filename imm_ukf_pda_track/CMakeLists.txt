cmake_minimum_required(VERSION 2.8.3)
project(imm_ukf_pda_track)

find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  geometry_msgs
  pcl_ros
  roscpp
  tf
)


set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

catkin_package(
  CATKIN_DEPENDS
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#imm_ukf_pda
add_executable(imm_ukf_pda
  nodes/imm_ukf_pda/imm_ukf_pda_main.cpp
  nodes/imm_ukf_pda/imm_ukf_pda.cpp
  nodes/imm_ukf_pda/ukf.cpp
)
target_link_libraries(imm_ukf_pda
  ${catkin_LIBRARIES}
)
add_dependencies(imm_ukf_pda
  ${catkin_EXPORTED_TARGETS}
)

install(
  TARGETS
    imm_ukf_pda
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)
