// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

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

    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);

    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);

    std::string calib_path = dataset_folder+"sequences/" + sequence_number + "/calib.txt";
    std::vector<std::vector<double>> calibs=read_calib(calib_path);
    Eigen::Matrix<double,3,4> P2;
    P2 <<calibs[2][0],calibs[2][1],calibs[2][2],calibs[2][3],
    calibs[2][4],calibs[2][5],calibs[2][6],calibs[2][7],
    calibs[2][8],calibs[2][9],calibs[2][10],calibs[2][11];
    Eigen::Matrix<double,4,4> Tr;
    Tr <<calibs[4][0],calibs[4][1],calibs[4][2],calibs[4][3],
    calibs[4][4],calibs[4][5],calibs[4][6],calibs[4][7],
    calibs[4][8],calibs[4][9],calibs[4][10],calibs[4][11],
    0,0,0,1;

    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);

    std::string line;
    std::size_t line_num = 0;

    ros::Rate r(10.0 / publish_delay);
    while (std::getline(timestamp_file, line) && ros::ok())
    {
        float timestamp = stof(line);
        std::stringstream left_image_path;
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_2/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat left_image = cv::imread(left_image_path.str());

        // read lidar point cloud
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        // std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";

        std::vector<Eigen::Vector3d> lidar_points;
        std::vector<float> lidar_intensities;
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);
            // if(lidar_data[i]>=0)
            // {
            // Eigen::Vector4d P_xyz(lidar_data[i],lidar_data[i + 1],lidar_data[i + 2],1);
            // Eigen::Vector3d P_uv = P2*Tr*P_xyz;
            // P_uv << P_uv[0]/P_uv[2],P_uv[1]/P_uv[2],1;
            //     if(P_uv[0]>=0 && P_uv[0]<=left_image.size().width && P_uv[1]>=0 && P_uv[1]<=left_image.size().height)
            //     {
            //         pcl::PointXYZI point;
            //         point.x = lidar_data[i];
            //         point.y = lidar_data[i + 1];
            //         point.z = lidar_data[i + 2];
            //         point.intensity = lidar_data[i + 3];
            //         laser_cloud.push_back(point);
            //     }
            // }
                    pcl::PointXYZI point;
                    point.x = lidar_data[i];
                    point.y = lidar_data[i + 1];
                    point.z = lidar_data[i + 2];
                    point.intensity = lidar_data[i + 3];
                    laser_cloud.push_back(point);
        }
        std::cout << "totally " <<lidar_data.size() / 4.0<<"after crop"<< laser_cloud.size()  << " points  \n";


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

        

        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "/velodyne";
        pub_laser_cloud.publish(laser_cloud_msg);

        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "bgr8", left_image).toImageMsg();
        pub_image_left.publish(image_left_msg);

        if (to_bag)
        {
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/points_raw", ros::Time::now(), laser_cloud_msg);
        }

        line_num ++;
        r.sleep();
    }
    bag_out.close();
    std::cout << "Done \n";


    return 0;
}