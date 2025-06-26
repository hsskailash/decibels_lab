#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/colors.h>
#include <memory>
#include <string>
#include <vector>
#include <random>

// Include sensor-specific point types
#include "os_point.h"        // Ouster point type
#include "velodyne_point.h"  // Velodyne point type

using PointT = pcl::PointXYZ;  // Use XYZ for clustering
using CloudT = pcl::PointCloud<PointT>;
using ColorPointT = pcl::PointXYZRGB;
using ColorCloudT = pcl::PointCloud<ColorPointT>;

class EuclideanClusteringNode : public rclcpp::Node {
public:
    EuclideanClusteringNode() : Node("euclidean_clustering") {
        // Declare parameters with defaults
        this->declare_parameter<std::string>("sensor_type", "velodyne");
        this->declare_parameter<std::string>("input_topic", "non_ground_points");
        this->declare_parameter<std::string>("colored_topic", "colored_clusters");
        this->declare_parameter<double>("cluster_tolerance", 0.5);
        this->declare_parameter<int>("min_cluster_size", 50);
        this->declare_parameter<int>("max_cluster_size", 5000);

        // Get parameters
        sensor_type_ = this->get_parameter("sensor_type").as_string();
        input_topic_ = this->get_parameter("input_topic").as_string();
        colored_topic_ = this->get_parameter("colored_topic").as_string();
        cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
        min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();

        // Validate sensor type
        if (sensor_type_ != "velodyne" && sensor_type_ != "ouster") {
            RCLCPP_ERROR(this->get_logger(), "Invalid sensor type: %s. Use 'velodyne' or 'ouster'", 
                         sensor_type_.c_str());
            rclcpp::shutdown();
            return;
        }

        // Create publisher/subscriber
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::SensorDataQoS(),
            std::bind(&EuclideanClusteringNode::cloud_callback, this, std::placeholders::_1));
        
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(colored_topic_, rclcpp::SensorDataQoS());

        RCLCPP_INFO(this->get_logger(), "Euclidean clustering node ready");
        RCLCPP_INFO(this->get_logger(), "  Input topic: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Cluster tolerance: %.2f m", cluster_tolerance_);
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        // Convert to PCL XYZ format
        CloudT::Ptr cloud(new CloudT);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud!");
            return;
        }

        // Perform clustering
        auto cluster_indices = extract_clusters(cloud);
        
        // Create colored cloud
        auto colored_cloud = colorize_clusters(cloud, cluster_indices);
        
        // Publish colored cloud
        publish_colored_cloud(colored_cloud, msg->header);
    }

    std::vector<pcl::PointIndices> extract_clusters(const CloudT::Ptr& cloud) {
        // Create KD-tree for efficient search
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);

        // Euclidean clustering
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        RCLCPP_INFO(this->get_logger(), "Found %zu clusters", cluster_indices.size());
        return cluster_indices;
    }

    ColorCloudT::Ptr colorize_clusters(
        const CloudT::Ptr& input_cloud,
        const std::vector<pcl::PointIndices>& cluster_indices) 
    {
        // Initialize colored cloud (all points white by default)
        ColorCloudT::Ptr colored_cloud(new ColorCloudT);
        colored_cloud->resize(input_cloud->size());
        
        // Copy XYZ data
        for (size_t i = 0; i < input_cloud->size(); ++i) {
            colored_cloud->points[i].x = input_cloud->points[i].x;
            colored_cloud->points[i].y = input_cloud->points[i].y;
            colored_cloud->points[i].z = input_cloud->points[i].z;
            colored_cloud->points[i].r = 255;  // Default white
            colored_cloud->points[i].g = 255;
            colored_cloud->points[i].b = 255;
        }

        // Random color generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint8_t> color_dist(50, 255);

        // Assign colors to clusters
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            // Generate vibrant color
            uint8_t r = color_dist(gen);
            uint8_t g = color_dist(gen);
            uint8_t b = color_dist(gen);

            // Apply to all points in cluster
            for (const auto& idx : cluster_indices[i].indices) {
                auto& point = colored_cloud->points[idx];
                point.r = r;
                point.g = g;
                point.b = b;
                // RCLCPP_INFO(this->get_logger(), "colors r=%d, g=%d, b=%d", r,g,b);
            }
        }

        return colored_cloud;
    }

    void publish_colored_cloud(
        const ColorCloudT::Ptr& cloud,
        const std_msgs::msg::Header& header) 
    {
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*cloud, output);
        output.header = header;
        pub_->publish(output);
    }

    // ROS components
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    
    // Parameters
    std::string sensor_type_;
    std::string input_topic_;
    std::string colored_topic_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EuclideanClusteringNode>());
    rclcpp::shutdown();
    return 0;
}