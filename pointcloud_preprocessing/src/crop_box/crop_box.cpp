#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "filter.hpp"



class CropBoxFilter : public Filter {
public:
    CropBoxFilter(std::string node_name, const rclcpp::NodeOptions & options) 
        : Filter("crop_box_filter", options) 
    {
        // Subscriber for input point cloud
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic + "/ground_filtered", rclcpp::SensorDataQoS(), 
            std::bind(&CropBoxFilter::pointcloudCallback, this, std::placeholders::_1));

        // Publisher for filtered point cloud
        filtered_pointcloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic + "/filtered", rclcpp::SensorDataQoS());

        RCLCPP_INFO(get_logger(), "CropBoxFilter node initialized.");
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) override {
        if (sensor == SensorType::OUSTER) {
            auto cloud = std::make_shared<OusterPC>();
            pcl::fromROSMsg(*msg, *cloud);
            input_cloud_ = cloud;
            
        } 

        
        // Perform the crop filter operation
        crop_filter();
    }

    void crop_filter() {
        // Lookup transform between robot_base_frame and lidar_frame
        geometry_msgs::msg::TransformStamped transformStamped;
        try {
            auto tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
            tf2_ros::TransformListener tfListener(*tfBuffer);

            transformStamped = tfBuffer->lookupTransform(
                robot_base_frame, lidarFrame, tf2::TimePointZero, tf2::durationFromSec(1.0));
        } catch (tf2::TransformException & ex) {
            RCLCPP_ERROR(get_logger(), "Transform lookup failed: %s", ex.what());
            return;
        }

        // Extract translation values
        double tx = transformStamped.transform.translation.x;
        double ty = transformStamped.transform.translation.y;
        double tz = transformStamped.transform.translation.z;

        // RCLCPP_INFO(get_logger(), "lidar position: x = %.2f, y = %.2f, z = %.2f", tx, ty, tz);

        // Calculate bounding box limits
        double min_x = -tx - length / 2;
        double max_x = -tx + length / 2;
        double min_y = -ty - width / 2;
        double max_y = -ty + width / 2;
        double max_z = -tz + height;
        double min_z = -tz-1.0;

        // RCLCPP_INFO(get_logger(), "box dimensions: min_x = %.2f, min_y = %.2f, min_z = %.2f, max_x = %.2f, max_y = %.2f, max_z = %.2f", min_x, min_y, min_z, max_x, max_y, max_z);

        // Determine the sensor type and cast input_cloud_
        if (sensor == SensorType::OUSTER) {
            auto cloud = std::static_pointer_cast<OusterPC>(input_cloud_);
            auto filtered_cloud = std::make_shared<OusterPC>();

            // Remove NaN points
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

            // Filter points
            for (const auto & point : cloud->points) {
                bool in_box = point.x >= min_x && point.x <= max_x &&
                              point.y >= min_y && point.y <= max_y &&
                              point.z >= min_z && point.z <= max_z;

                if ((negative && in_box) || (!negative && !in_box)) {
                    filtered_cloud->points.push_back(point);
                }
            }

            // Publish the filtered point cloud
            filtered_cloud->width = filtered_cloud->points.size();
            filtered_cloud->height = 1;
            filtered_cloud->is_dense = true;

            sensor_msgs::msg::PointCloud2 output_msg;
            
        
        
            pcl::toROSMsg(*filtered_cloud, output_msg);
            std_msgs::msg::Header header;
            header.frame_id = lidarFrame;
            header.stamp = get_clock()->now();
            output_msg.header = header;
            filtered_pointcloud_pub_->publish(output_msg);

        } 
        // else if (sensor == SensorType::VELODYNE) {
        //     auto cloud = std::static_pointer_cast<VelodynePC>(input_cloud_);
        //     auto filtered_cloud = std::make_shared<VelodynePC>();

        //     // Remove NaN points
        //     std::vector<int> indices;
        //     pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

        //     // Filter points
        //     for (const auto & point : cloud->points) {
        //         bool in_box = point.x >= min_x && point.x <= max_x &&
        //                       point.y >= min_y && point.y <= max_y &&
        //                       point.z >= min_z && point.z <= max_z;

        //         if ((negative && in_box) || (!negative && !in_box)) {
        //             filtered_cloud->points.push_back(point);
        //         }
        //     }

        //     // Publish the filtered point cloud
        //     filtered_cloud->width = filtered_cloud->points.size();
        //     filtered_cloud->height = 1;
        //     filtered_cloud->is_dense = true;

        //     sensor_msgs::msg::PointCloud2 output_msg;
        //     pcl::toROSMsg(*filtered_cloud, output_msg);
        //     output_msg.header.frame_id = lidarFrame;
        //     filtered_pointcloud_pub_->publish(output_msg);
        // }
    }

    
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    try {
        auto crop_box_filter_node = std::make_shared<CropBoxFilter>("crop_box_filter", rclcpp::NodeOptions());
        rclcpp::spin(crop_box_filter_node);
    } catch (const std::exception &e) {
        std::cerr << "Error occurred while running CropBoxFilter node: " << e.what() << std::endl;
    }

    rclcpp::shutdown();
    return 0;
}
