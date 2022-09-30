// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <stdint.h>
#include <unordered_map>

#include <python2.7/Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "aloam_velodyne/tic_toc.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

namespace std {
  template <>
  class hash< cv::Point >{
  public :
    size_t operator()(const cv::Point &pixel_cloud ) const
    {
      return hash<std::string>()( std::to_string(pixel_cloud.x) + "|" + std::to_string(pixel_cloud.y) );
    }
  };
};

void init_numpy()
{
    import_array();
}
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

std::vector<std::vector<double>> read_calib(const std::string calib_path)
{
    std::ifstream calib_file(calib_path,std::ifstream::in);
    std::string s;
    std::string line;
    std::vector<double> calib_matrix;
    std::vector<std::vector<double>> calibs;
    while(std::getline(calib_file, line))
    {
    std::stringstream calib_stream(line);
    std::getline(calib_stream, s, ' ');
    for (std::size_t i = 0; i < 12; ++i)
    {
        std::getline(calib_stream, s, ' ');
        calib_matrix.push_back(stod(s));
    }
    calibs.push_back(calib_matrix);
    calib_matrix.clear();
    }
    return calibs;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kitti_helper");

    ros::NodeHandle n("~");
    std::string dataset_folder, sequence_number, output_bag_file;
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence_number);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    n.getParam("to_bag", to_bag);
    if (to_bag)
    {
        n.getParam("output_bag_file", output_bag_file);
        output_bag_file += std::string("kitti_")+sequence_number+std::string("_track.bag");
        std::cout<<output_bag_file<<std::endl;
    }
    int publish_delay;
    n.getParam("publish_delay", publish_delay);
    publish_delay = publish_delay <= 0 ? 1 : publish_delay;

    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/points_raw", 2);
    ros::Publisher pub_camera_info = n.advertise<sensor_msgs::CameraInfo>("/camera_info", 2);
    ros::Publisher pub_fuse_cloud = n.advertise<sensor_msgs::PointCloud2>("/points_fuse", 2);
    ros::Publisher pub_dense_cloud_rgb = n.advertise<sensor_msgs::PointCloud2>("/points_dense", 2);

    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_depth = it.advertise("/image_depth", 2);
    image_transport::Publisher pub_image_depth_dense = it.advertise("/image_depth_dense", 2);

    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string calib_path = dataset_folder+"sequences/" + sequence_number + "/calib.txt";
    std::vector<std::vector<double>> calibs=read_calib(calib_path);

    Eigen::Matrix<double,3,4> P2;
    P2 <<calibs[2][0],calibs[2][1],calibs[2][2],calibs[2][3],
    calibs[2][4],calibs[2][5],calibs[2][6],calibs[2][7],
    calibs[2][8],calibs[2][9],calibs[2][10],calibs[2][11];

    Eigen::Matrix<double,4,4> T2;
    T2 <<1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,P2(0,3)/P2(0,0);

    Eigen::Matrix<double,4,4> Tr;
    Tr <<calibs[4][0],calibs[4][1],calibs[4][2],calibs[4][3],
    calibs[4][4],calibs[4][5],calibs[4][6],calibs[4][7],
    calibs[4][8],calibs[4][9],calibs[4][10],calibs[4][11],
    0,0,0,1;

    Eigen::Isometry3d T_cam2_velo=  Eigen::Isometry3d::Identity();  
    // Eigen::Matrix<double,4,4> T_cam2_velo;
    T_cam2_velo=T2*Tr;

      Eigen::Isometry3d::Identity();

    Eigen::Matrix<double,3,4> project_matrix;
    project_matrix= P2*Tr;

    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);

    std::string line;
    std::size_t line_num = 0;

    ros::Rate r(10.0 / publish_delay);

    Py_Initialize();
    init_numpy();
    PyRun_SimpleString("import os");

    PyRun_SimpleString("import sys");
    // PyRun_SimpleString("sys.path.append(os.path.abspath(os.path.join(os.getcwd(), \"..\"))+'/python/')");
//    PyRun_SimpleString("sys.path.append('/home/wenyu/catkin_ws/src/SLAMMOT/A-LOAM/src')"); 

    PyRun_SimpleString("sys.path.append('/home/wenyu/catkin_ws/src/SLAMMOT/A-LOAM/src')"); 
    
    PyObject* pModule =  PyImport_Import(PyString_FromString("depth_map_utils"));
    // PyObject* pModule =  PyImport_ImportModule("depth_map_utils");

    if (pModule == nullptr)
    {
        std::cout<<"error loading"<<std::endl;
        PyErr_Print();
        std::exit(1);
    }
    // PyObject* pFunc =PyObject_GetAttrString(pModule, "fill_in_multiscale");

    PyObject *pDict    =  PyModule_GetDict(pModule);
    PyObject* pFunc =PyDict_GetItemString(pDict, "fill_in_multiscale");

    PyObject* pArgs = PyTuple_New(1);
    std::cout<<"load funs"<<std::endl;

    while (std::getline(timestamp_file, line) && ros::ok())
    {
        TicToc t_whole;
        float timestamp = stof(line);

        std::stringstream left_image_path;
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_2/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat left_image = cv::imread(left_image_path.str());

        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());

        std::vector<Eigen::Vector3d> lidar_points;
        std::vector<float> lidar_intensities;
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;
        pcl::PointCloud<pcl::PointXYZRGB> laser_cloud_RGB;
        std::unordered_map<cv::Point, pcl::PointXYZ> projection_map;

        cv::Mat depth_image = cv::Mat(left_image.size().height, left_image.size().width,  CV_16UC1,cv::Scalar(0));

        uint16_t depth_image_array[left_image.size().height][left_image.size().width]={0};
        std::cout<<depth_image_array[left_image.size().height-1][left_image.size().width-1]<<std::endl;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            // lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            // lidar_intensities.push_back(lidar_data[i+3]);

            Eigen::Vector4d P_xyz(lidar_data[i],lidar_data[i + 1],lidar_data[i + 2],1);
            Eigen::Vector3d P_uv = project_matrix*P_xyz;
            P_uv << P_uv[0]/P_uv[2],P_uv[1]/P_uv[2],P_uv[2];

            int u = int(P_uv[0]);
            int v = int(P_uv[1]);

                if(u>=0 && u<left_image.size().width && v>=0 && v<left_image.size().height && lidar_data[i]>0)
                {
                    // pcl::PointXYZ point_1;
                    // point_1.x = lidar_data[i];
                    // point_1.y = lidar_data[i + 1];
                    // point_1.z = lidar_data[i + 2];
                    // projection_map.insert(std::pair<cv::Point, pcl::PointXYZ>(cv::Point(u, v), point_1));
                    cv::Vec3b rgb_pixel = left_image.at<cv::Vec3b>(v, u);
                    pcl::PointXYZRGB colored_3d_point;
                    colored_3d_point.x = lidar_data[i];
                    colored_3d_point.y = lidar_data[i + 1];
                    colored_3d_point.z = lidar_data[i + 2];
                    colored_3d_point.r = rgb_pixel[2];
                    colored_3d_point.g = rgb_pixel[1];
                    colored_3d_point.b = rgb_pixel[0];
                    laser_cloud_RGB.push_back(colored_3d_point);
                    depth_image.at<uint16_t>(v, u) =  uint16_t(P_uv[2]*256) ; 
                    depth_image_array[v][u]= uint16_t(P_uv[2]*256) ; 
                }

                    pcl::PointXYZI point;
                    point.x = lidar_data[i];
                    point.y = lidar_data[i + 1];
                    point.z = lidar_data[i + 2];
                    point.intensity = lidar_data[i + 3];
                    laser_cloud.push_back(point);
        }

        npy_intp Dims[2] = { left_image.size().height,left_image.size().width}; 

        PyObject *PyArray  = PyArray_SimpleNewFromData(2, Dims, NPY_UINT16, depth_image.data); 

        PyTuple_SetItem(pArgs, 0, PyArray); 
        PyObject* pReturn = PyObject_CallObject(pFunc, pArgs);
        uint16_t *data = (uint16_t *)PyByteArray_AsString(pReturn );
        cv::Mat dense_depth_image(left_image.size().height,  left_image.size().width, CV_16UC1, data);

        pcl::PointCloud<pcl::PointXYZRGB> laser_cloud_dense;
        // Eigen::Matrix<double,4,4> T_cam2_velo_inv;
        Eigen::Isometry3d T_cam2_velo_inv=  T_cam2_velo.inverse();
        // T_cam2_velo_inv.rotate(T_cam2_velo.block(0,0,3,3));
        // T_cam2_velo_inv.translate(T_cam2_velo.block(0,3,3,1));
        // std::cout << T_cam2_velo_inv.matrix() << std::endl;

        for(auto cv_row = 0; cv_row < dense_depth_image.size().height; cv_row++){
            for(auto cv_col = 0; cv_col < dense_depth_image.size().width; cv_col++){
                    pcl::PointXYZRGB dense_3d_point;
                    cv::Vec3b rgb_pixel = left_image.at<cv::Vec3b>(cv_row, cv_col);


                    Eigen::Vector4d P_cam(
                        uint16_t(dense_depth_image.at<uint16_t>(cv_row, cv_col)/256)*(cv_col-P2(0,2))/P2(0,0),
                        uint16_t(dense_depth_image.at<uint16_t>(cv_row, cv_col)/256)*(cv_row-P2(1,2))/P2(1,1),
                        uint16_t(dense_depth_image.at<uint16_t>(cv_row, cv_col)/256),
                        1);
                    Eigen::Vector4d P_velo = T_cam2_velo_inv*P_cam;
                    dense_3d_point.x = P_velo(0);
                    dense_3d_point.y = P_velo(1);;
                    dense_3d_point.z = P_velo(2);;
                    dense_3d_point.r = rgb_pixel[2];
                    dense_3d_point.g = rgb_pixel[1];
                    dense_3d_point.b = rgb_pixel[0];
                    laser_cloud_dense.push_back(dense_3d_point);
            }
        }


        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "/velodyne";
        pub_laser_cloud.publish(laser_cloud_msg);

        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(laser_cloud_RGB, cloud_msg);
        cloud_msg.header = laser_cloud_msg.header;
        pub_fuse_cloud.publish(cloud_msg);

        sensor_msgs::PointCloud2 cloud_msg_dense_rgb;
        pcl::toROSMsg(laser_cloud_dense, cloud_msg_dense_rgb);
        cloud_msg_dense_rgb.header.stamp = ros::Time().fromSec(timestamp);
        cloud_msg_dense_rgb.header.frame_id = "/velodyne";
        pub_dense_cloud_rgb.publish(cloud_msg_dense_rgb);

        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "bgr8", left_image).toImageMsg();
        pub_image_left.publish(image_left_msg);

        depth_image.convertTo(depth_image,CV_16UC1);
        sensor_msgs::ImagePtr image_depth_msg = cv_bridge::CvImage(laser_cloud_msg.header, "16UC1", depth_image).toImageMsg();
        pub_image_depth.publish(image_depth_msg);

        dense_depth_image.convertTo(dense_depth_image,CV_16UC1);
        sensor_msgs::ImagePtr image_depth_dense_msg = cv_bridge::CvImage(laser_cloud_msg.header, "16UC1", dense_depth_image).toImageMsg();
        pub_image_depth_dense.publish(image_depth_dense_msg);

        sensor_msgs::CameraInfo lidar2image;
        lidar2image.header.stamp = ros::Time().fromSec(timestamp);
        lidar2image.header.frame_id = "/camera_init";

        lidar2image.P={T_cam2_velo(0,0),T_cam2_velo(0,1),T_cam2_velo(0,2),T_cam2_velo(0,3),
                                        T_cam2_velo(1,0),T_cam2_velo(1,1),T_cam2_velo(1,2),T_cam2_velo(1,3),
                                        T_cam2_velo(2,0),T_cam2_velo(2,1),T_cam2_velo(2,2),T_cam2_velo(2,3)};

        lidar2image.K={P2(0,0),P2(0,1),P2(0,2),
                                         P2(1,0),P2(1,1),P2(1,2),
                                         P2(2,0),P2(2,1),P2(2,2)};
        lidar2image.height=left_image.size().height;
        lidar2image.width=left_image.size().width;
        pub_camera_info.publish(lidar2image);
        
        if (to_bag)
        {
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_depth", ros::Time::now(), image_depth_msg);
            bag_out.write("/points_raw", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/points_fuse", ros::Time::now(), cloud_msg);
            bag_out.write("/camera_info", ros::Time::now(), lidar2image);

        }
        std::cout<<"frame:"<<line_num<<std::endl;
        line_num ++;
        printf("process dataset %f ms *************\n", t_whole.toc());
        r.sleep();
    }
    bag_out.close();
    std::cout << "Done \n";
    Py_Finalize();

    return 0;
}    

////save pcd to bin file
    // std::ofstream out;
    // std::stringstream save_filename;
    // save_filename << dataset_folder << "sequences/" + sequence_number + "/fv/" 
    //                 << std::setfill('0') << std::setw(6) << line_num << ".bin";
    // out.open(save_filename.str(), std::ios::out | std::ios::binary);
    // std::cout << save_filename.str() << " saved" << std::endl;
    // int cloudSize = laser_cloud.points.size();
    // for (int i = 0; i < cloudSize; ++i)
    // {
    //     float point_x = laser_cloud.points[i].x;
    //     float point_y = laser_cloud.points[i].y;
    //     float point_z = laser_cloud.points[i].z;
    //     out.write(reinterpret_cast<const char *>(&point_x), sizeof(float));
    //     out.write(reinterpret_cast<const char *>(&point_y), sizeof(float));
    //     out.write(reinterpret_cast<const char *>(&point_z), sizeof(float));
    // }
    // out.close();
