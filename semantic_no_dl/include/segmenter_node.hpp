#ifndef OUSTER_SEGMENTER__SEGMENTER_NODE_HPP_
#define OUSTER_SEGMENTER__SEGMENTER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <string>
#include <map>

class LidarSegmenter : public rclcpp::Node {
public:
    LidarSegmenter();

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocess(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>
    segment_ground(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    std::vector<pcl::PointIndices> cluster_points(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    std::tuple<pcl::PointCloud<pcl::PointXYZRGB>, std::vector<int>> classify_objects(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
        const std::vector<pcl::PointIndices>& clusters);
    
    void publish_results(
        const pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
        const std::vector<pcl::PointIndices>& clusters,
        const std::vector<int>& cluster_labels);

    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    
    // Parameters
    double voxel_size_;
    double ground_threshold_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    double z_min_;
    double z_max_;

    // Label mapping
    std::map<int, std::string> label_map_;
    std::map<int, std::vector<float>> color_map_;
};

#endif  // OUSTER_SEGMENTER__SEGMENTER_NODE_HPP_