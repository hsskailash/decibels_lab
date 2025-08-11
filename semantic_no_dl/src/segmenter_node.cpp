#include "ouster_segmenter/segmenter_node.hpp"
#include <pcl/common/centroid.h>
#include <algorithm>

using namespace std::chrono_literals;

enum ClassLabel {
    UNCLASSIFIED = 0,
    GROUND = 1,
    VEGETATION = 2,
    BUILDING = 3,
    VEHICLE = 4,
    POLE = 5
};

LidarSegmenter::LidarSegmenter() : Node("ouster_segmenter") {
    // Parameters
    declare_parameter("voxel_size", 0.1);
    declare_parameter("ground_threshold", 0.3);
    declare_parameter("cluster_tolerance", 0.5);
    declare_parameter("min_cluster_size", 10);
    declare_parameter("max_cluster_size", 10000);

    voxel_size_ = get_parameter("voxel_size").as_double();
    ground_threshold_ = get_parameter("ground_threshold").as_double();
    cluster_tolerance_ = get_parameter("cluster_tolerance").as_double();
    min_cluster_size_ = get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = get_parameter("max_cluster_size").as_int();

    // Subscriber and Publishers
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ouster/points", 10,
        std::bind(&LidarSegmenter::cloud_callback, this, std::placeholders::_1));
    
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/segmented/points", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/segmented/markers", 10);

    RCLCPP_INFO(get_logger(), "Ouster Segmenter initialized");
}

void LidarSegmenter::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto start = this->now();
    
    // Convert to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) return;

    // Processing pipeline
    auto filtered = preprocess(cloud);
    if (filtered->empty()) return;

    auto [ground, non_ground] = segment_ground(filtered);
    auto clusters = cluster_points(non_ground);
    auto colored_cloud = classify_objects(ground, non_ground, clusters);
    publish_results(colored_cloud, non_ground, clusters);
    
    auto duration = (this->now() - start).seconds();
    RCLCPP_DEBUG(get_logger(), "Segmented %zu objects in %.3f sec", clusters.size(), duration);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarSegmenter::preprocess(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Voxel Downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(cloud);
    voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel.filter(*filtered);

    // Statistical Outlier Removal
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered);
    sor.setMeanK(20);
    sor.setStddevMulThresh(2.0);
    sor.filter(*filtered);

    return filtered;
}

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr>
LidarSegmenter::segment_ground(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground(new pcl::PointCloud<pcl::PointXYZ>);

    // Find min height
    float min_z = std::numeric_limits<float>::max();
    for (const auto& pt : *cloud) min_z = std::min(min_z, pt.z);

    // Height thresholding
    const float threshold = min_z + ground_threshold_;
    for (const auto& pt : *cloud) {
        if (pt.z < threshold) ground->push_back(pt);
        else non_ground->push_back(pt);
    }

    return {ground, non_ground};
}

std::vector<pcl::PointIndices> LidarSegmenter::cluster_points(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) 
{
    if (cloud->empty()) return {};

    // Create KD-Tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    // Euclidean Clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    return cluster_indices;
}

pcl::PointCloud<pcl::PointXYZRGB> LidarSegmenter::classify_objects(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
    const std::vector<pcl::PointIndices>& clusters) 
{
    pcl::PointCloud<pcl::PointXYZRGB> result;

    // Color ground points (green)
    for (const auto& pt : *ground) {
        pcl::PointXYZRGB color_pt;
        color_pt.x = pt.x;
        color_pt.y = pt.y;
        color_pt.z = pt.z;
        color_pt.r = 50;
        color_pt.g = 200;
        color_pt.b = 50;
        result.push_back(color_pt);
    }

    // Classify clusters
    for (const auto& indices : clusters) {
        // Extract cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
        for (auto idx : indices.indices) cluster->push_back((*non_ground)[idx]);

        // Compute features
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);
        Eigen::Vector3f dimensions = max_pt.head<3>() - min_pt.head<3>();

        // PCA for shape analysis
        pcl::PCA<pcl::PointXYZ> pca(true);
        pca.setInputCloud(cluster);
        Eigen::Vector3f eigenvalues = pca.getEigenValues();
        if (eigenvalues[0] < 1e-6) continue;  // Skip invalid clusters
        
        float lambda1 = eigenvalues[0], lambda2 = eigenvalues[1], lambda3 = eigenvalues[2];
        float linearity = (lambda1 - lambda2) / lambda1;
        float planarity = (lambda2 - lambda3) / lambda1;
        float sphericity = lambda3 / lambda1;

        // Classify based on rules
        ClassLabel label = UNCLASSIFIED;
        if (dimensions.z() > 3.0 && planarity > 0.7) {
            label = BUILDING;  // Red
        } else if (dimensions.z() > 2.0 && dimensions.x() < 0.5 && dimensions.y() < 0.5) {
            label = POLE;      // Yellow
        } else if (dimensions.z() > 1.0 && dimensions.z() < 2.5 && 
                   dimensions.x() > 1.5 && dimensions.x() < 5.0 &&
                   dimensions.y() > 1.0 && dimensions.y() < 2.5) {
            label = VEHICLE;   // Blue
        } else if (sphericity > 0.4 && planarity < 0.3) {
            label = VEGETATION; // Bright green
        }

        // Color points based on class
        for (auto idx : indices.indices) {
            pcl::PointXYZRGB color_pt;
            color_pt.x = (*non_ground)[idx].x;
            color_pt.y = (*non_ground)[idx].y;
            color_pt.z = (*non_ground)[idx].z;

            switch(label) {
                case BUILDING:
                    color_pt.r = 255; color_pt.g = 50; color_pt.b = 50; break;
                case VEHICLE:
                    color_pt.r = 50; color_pt.g = 50; color_pt.b = 255; break;
                case VEGETATION:
                    color_pt.r = 50; color_pt.g = 255; color_pt.b = 50; break;
                case POLE:
                    color_pt.r = 255; color_pt.g = 255; color_pt.b = 0; break;
                default:  // Unclassified
                    color_pt.r = 128; color_pt.g = 128; color_pt.b = 128;
            }
            result.push_back(color_pt);
        }
    }

    return result;
}

void LidarSegmenter::publish_results(
    const pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
    const std::vector<pcl::PointIndices>& clusters) 
{
    // Publish colored point cloud
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(colored_cloud, output);
    output.header.frame_id = "os_sensor";
    output.header.stamp = now();
    pub_->publish(output);

    // Create bounding box markers
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_marker.header.frame_id = "os_sensor";
    clear_marker.header.stamp = now();
    markers.markers.push_back(clear_marker);

    int id = 0;
    for (const auto& indices : clusters) {
        if (indices.indices.size() < 3) continue;

        // Calculate bounding box
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*non_ground, indices.indices, min_pt, max_pt);

        // Create marker
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "os_sensor";
        marker.header.stamp = now();
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::CUBE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = (min_pt.x() + max_pt.x()) / 2;
        marker.pose.position.y = (min_pt.y() + max_pt.y()) / 2;
        marker.pose.position.z = (min_pt.z() + max_pt.z()) / 2;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = max_pt.x() - min_pt.x();
        marker.scale.y = max_pt.y() - min_pt.y();
        marker.scale.z = max_pt.z() - min_pt.z();
        
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 0.3f;
        
        marker.lifetime = rclcpp::Duration(0.1s);
        markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarSegmenter>());
    rclcpp::shutdown();
    return 0;
}