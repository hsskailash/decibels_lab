#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/kdtree/kdtree_flann.h>  // pcl include kdtree_flann throws error if PCL_NO_PRECOMPILE
                                      // is defined before
#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include "os_point.h"

// struct VelodynePointXYZIRT {
//         PCL_ADD_POINT4D;
//         PCL_ADD_INTENSITY;
//         uint16_t ring;
//         float time;
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//     } EIGEN_ALIGN16;

// POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePointXYZIRT,
//                                   (float, x, x)
//                                   (float, y, y)
//                                   (float, z, z)
//                                   (float, intensity, intensity)
//                                   (uint16_t, ring, ring)
//                                   (float, time, time));

// using VelodynePC = VelodynePointXYZIRT;
using OusterPC = pcl::PointCloud<ouster_ros::Point>;
enum class SensorType { VELODYNE, OUSTER};

class Filter : public rclcpp::Node {
protected:
    // Member variables for configuration
    std::string pointCloudTopic;
    std::string lidarFrame;
    std::string robot_base_frame;
    float length;
    float width;
    float height;
    bool negative;
    
    // Input and output point clouds as smart pointers
    std::shared_ptr<void> input_cloud_;
    std::shared_ptr<void> output_cloud_;

    // Sensor type (Velodyne or Ouster)
    SensorType sensor = SensorType::OUSTER;

    // ROS2 Subscriber and Publisher
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pointcloud_pub_;


    

public:
    Filter(const std::string& node_name, const rclcpp::NodeOptions& options)
        : Node(node_name, options), input_cloud_(nullptr), output_cloud_(nullptr) {
        // Declare and get ROS parameters
        declare_parameter("pointCloudTopic", "points");
        get_parameter("pointCloudTopic", pointCloudTopic);

        declare_parameter("robot_base_frame", "base_link");
        get_parameter("robot_base_frame", robot_base_frame);

        declare_parameter("lidarFrame", "os_sensor");
        get_parameter("lidarFrame", lidarFrame);

        declare_parameter("sensor", "ouster");
        std::string sensorStr;
        get_parameter("sensor", sensorStr);
        // sensor = (sensorStr == "velodyne") ? SensorType::VELODYNE : SensorType::OUSTER;
        if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        // else if (sensorStr == "velodyne")
        // {
        //     sensor = SensorType::VELODYNE;
        // }
        
        else
        {
            RCLCPP_ERROR_STREAM(
                get_logger(),
                "Invalid sensor type (must be 'ouster' ): " << sensorStr);
            rclcpp::shutdown();
        }

        declare_parameter("length", 0.0);
        get_parameter("length", length);
        declare_parameter("width", 0.0);
        get_parameter("width", width);
        declare_parameter("height", 0.0);
        get_parameter("height", height);
        declare_parameter("negative", false);
        get_parameter("negative", negative);

        // Set up ROS2 subscriber and publisher
      
    }

    // Virtual destructor for cleanup
    virtual ~Filter() = default;

    // Accessors for point clouds
    std::shared_ptr<void> getInputCloud() const { return input_cloud_; }
    std::shared_ptr<void> getOutputCloud() const { return output_cloud_; }

    // Virtual callback function for point cloud processing
    virtual void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (sensor == SensorType::OUSTER) {
            auto cloud = std::make_shared<OusterPC>();
            pcl::fromROSMsg(*msg, *cloud);
            input_cloud_ = cloud;

            auto output_cloud = std::make_shared<OusterPC>();
            output_cloud_ = output_cloud;

        // } else if (sensor == SensorType::VELODYNE) {
        //     auto cloud = std::make_shared<VelodynePC>();
        //     pcl::fromROSMsg(*msg, *cloud);
        //     input_cloud_ = cloud;

        //     auto output_cloud = std::make_shared<VelodynePC>();
        //     output_cloud_ = output_cloud;
        // }

        // Publish the output cloud
        sensor_msgs::msg::PointCloud2 output_msg;
        if (sensor == SensorType::OUSTER) {
            pcl::toROSMsg(*std::static_pointer_cast<OusterPC>(output_cloud_), output_msg);
        } 
        // else if (sensor == SensorType::VELODYNE) {
        //     pcl::toROSMsg(*std::static_pointer_cast<VelodynePC>(output_cloud_), output_msg);
        // }
        filtered_pointcloud_pub_->publish(output_msg);
    }
    }
};

#endif 