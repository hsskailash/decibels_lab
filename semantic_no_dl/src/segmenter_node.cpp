#include "segmenter_node.hpp"
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <algorithm>

using namespace std::chrono_literals;

// Class labels
const int UNCLASSIFIED = 0;
const int GROUND = 1;
const int VEGETATION = 2;
const int BUILDING = 3;
const int VEHICLE = 4;
const int POLE = 5;

LidarSegmenter::LidarSegmenter() : Node("ouster_segmenter") {
    // Initialize label and color maps
    label_map_ = {
        {UNCLASSIFIED, "unknown"},
        {GROUND, "ground"},
        {VEGETATION, "vegetation"},
        {BUILDING, "building"},
        {VEHICLE, "vehicle"},
        {POLE, "pole"}
    };
    
    color_map_ = {
        {UNCLASSIFIED, {0.5, 0.5, 0.5}},   // Gray
        {GROUND, {0.2, 0.6, 0.2}},          // Green
        {VEGETATION, {0.0, 0.8, 0.0}},      // Bright Green
        {BUILDING, {0.8, 0.2, 0.2}},        // Red
        {VEHICLE, {0.2, 0.2, 0.8}},         // Blue
        {POLE, {0.8, 0.8, 0.0}}             // Yellow
    };

    // Parameters
    declare_parameter("voxel_size", 0.1);
    declare_parameter("ground_threshold", 0.3);
    declare_parameter("cluster_tolerance", 0.5);
    declare_parameter("min_cluster_size", 10);
    declare_parameter("max_cluster_size", 10000);
    declare_parameter("z_min", -5.0);
    declare_parameter("z_max", 20.0);

    voxel_size_ = get_parameter("voxel_size").as_double();
    ground_threshold_ = get_parameter("ground_threshold").as_double();
    cluster_tolerance_ = get_parameter("cluster_tolerance").as_double();
    min_cluster_size_ = get_parameter("min_cluster_size").as_int();
    max_cluster_size_ = get_parameter("max_cluster_size").as_int();
    z_min_ = get_parameter("z_min").as_double();
    z_max_ = get_parameter("z_max").as_double();

    // Subscriber and Publishers
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/ouster/points",  rclcpp::SensorDataQoS(),
        std::bind(&LidarSegmenter::cloud_callback, this, std::placeholders::_1));
    
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/segmented/points", rclcpp::SensorDataQoS());
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
    
    if (clusters.empty()) return;
    
    auto [colored_cloud, cluster_labels] = classify_objects(ground, non_ground, clusters);
    publish_results(colored_cloud, non_ground, clusters, cluster_labels);
    
    auto duration = (this->now() - start).seconds();
    RCLCPP_DEBUG(get_logger(), "Segmented %zu objects in %.3f sec", clusters.size(), duration);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr LidarSegmenter::preprocess(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 1. Remove NaN and Inf points
    filtered->reserve(cloud->size());
    for (const auto& pt : *cloud) {
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
            filtered->push_back(pt);
        }
    }
    RCLCPP_DEBUG(get_logger(), "Removed %ld invalid points", 
                 cloud->size() - filtered->size());

    if (filtered->empty()) return filtered;

    // 2. Z-axis range filtering (remove distant points)
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(filtered);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_min_, z_max_);
    pass.filter(*filtered);

    // 3. Voxel Downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(filtered);
    voxel.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel.filter(*filtered);

    // 4. Statistical Outlier Removal
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

    // Improved ground segmentation using adaptive threshold
    std::vector<float> z_values;
    z_values.reserve(cloud->size());
    for (const auto& pt : *cloud) z_values.push_back(pt.z);
    std::sort(z_values.begin(), z_values.end());
    
    // Use 10th percentile as ground reference
    int index = static_cast<int>(z_values.size() * 0.1);
    float ground_ref = z_values[std::max(0, index)];
    const float threshold = ground_ref + ground_threshold_;

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

std::tuple<pcl::PointCloud<pcl::PointXYZRGB>, std::vector<int>> 
LidarSegmenter::classify_objects(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& ground,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
    const std::vector<pcl::PointIndices>& clusters) 
{
    pcl::PointCloud<pcl::PointXYZRGB> result;
    std::vector<int> cluster_labels;

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
        if (eigenvalues[0] < 1e-6) {
            cluster_labels.push_back(UNCLASSIFIED);
            continue;
        }
        
        float lambda1 = eigenvalues[0], lambda2 = eigenvalues[1], lambda3 = eigenvalues[2];
        float linearity = (lambda1 - lambda2) / lambda1;
        float planarity = (lambda2 - lambda3) / lambda1;
        float sphericity = lambda3 / lambda1;

        // Classify based on rules
        int label = UNCLASSIFIED;
        if (dimensions.z() > 3.0 && planarity > 0.7) {
            label = BUILDING;
        } else if (dimensions.z() > 2.0 && dimensions.x() < 0.5 && dimensions.y() < 0.5) {
            label = POLE;
        } else if (dimensions.z() > 1.0 && dimensions.z() < 2.5 && 
                   dimensions.x() > 1.5 && dimensions.x() < 5.0 &&
                   dimensions.y() > 1.0 && dimensions.y() < 2.5) {
            label = VEHICLE;
        } else if (sphericity > 0.4 && planarity < 0.3) {
            label = VEGETATION;
        }

        cluster_labels.push_back(label);

        // Get color from map
        auto color = color_map_.at(label);

        // Color points based on class
        for (auto idx : indices.indices) {
            pcl::PointXYZRGB color_pt;
            color_pt.x = (*non_ground)[idx].x;
            color_pt.y = (*non_ground)[idx].y;
            color_pt.z = (*non_ground)[idx].z;
            color_pt.r = static_cast<uint8_t>(color[0] * 255);
            color_pt.g = static_cast<uint8_t>(color[1] * 255);
            color_pt.b = static_cast<uint8_t>(color[2] * 255);
            result.push_back(color_pt);
        }
    }

    return {result, cluster_labels};
}

void LidarSegmenter::publish_results(
    const pcl::PointCloud<pcl::PointXYZRGB>& colored_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground,
    const std::vector<pcl::PointIndices>& clusters,
    const std::vector<int>& cluster_labels) 
{
    // Publish colored point cloud
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(colored_cloud, output);
    output.header.frame_id = "os_sensor";
    output.header.stamp = now();
    pub_->publish(output);

    // Create bounding box and label markers
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_marker.header.frame_id = "os_sensor";
    clear_marker.header.stamp = now();
    markers.markers.push_back(clear_marker);

    int id = 0;
    for (size_t i = 0; i < clusters.size(); i++) {
        const auto& indices = clusters[i];
        if (indices.indices.size() < 3) continue;
        
        if (i >= cluster_labels.size()) continue;
        int label = cluster_labels[i];

        // Calculate bounding box
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*non_ground, indices.indices, min_pt, max_pt);

        // Create bounding box marker
        visualization_msgs::msg::Marker bbox_marker;
        bbox_marker.header.frame_id = "os_sensor";
        bbox_marker.header.stamp = now();
        bbox_marker.id = id++;
        bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
        bbox_marker.action = visualization_msgs::msg::Marker::ADD;
        
        bbox_marker.pose.position.x = (min_pt.x() + max_pt.x()) / 2;
        bbox_marker.pose.position.y = (min_pt.y() + max_pt.y()) / 2;
        bbox_marker.pose.position.z = (min_pt.z() + max_pt.z()) / 2;
        bbox_marker.pose.orientation.w = 1.0;
        
        bbox_marker.scale.x = max_pt.x() - min_pt.x();
        bbox_marker.scale.y = max_pt.y() - min_pt.y();
        bbox_marker.scale.z = max_pt.z() - min_pt.z();
        
        auto color = color_map_.at(label);
        bbox_marker.color.r = color[0];
        bbox_marker.color.g = color[1];
        bbox_marker.color.b = color[2];
        bbox_marker.color.a = 0.3;
        
        bbox_marker.lifetime = rclcpp::Duration(0.1s);
        markers.markers.push_back(bbox_marker);

        // Create text label marker
        visualization_msgs::msg::Marker label_marker;
        label_marker.header = bbox_marker.header;
        label_marker.id = id++;
        label_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        label_marker.action = visualization_msgs::msg::Marker::ADD;
        
        label_marker.pose.position.x = bbox_marker.pose.position.x;
        label_marker.pose.position.y = bbox_marker.pose.position.y;
        label_marker.pose.position.z = max_pt.z() + 0.5;  // Above the bounding box
        label_marker.pose.orientation.w = 1.0;
        
        label_marker.scale.z = 0.5;  // Text height
        label_marker.color.r = 1.0;
        label_marker.color.g = 1.0;
        label_marker.color.b = 1.0;
        label_marker.color.a = 1.0;
        
        label_marker.text = label_map_.at(label);
        label_marker.lifetime = rclcpp::Duration(0.1s);
        markers.markers.push_back(label_marker);
    }

    marker_pub_->publish(markers);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarSegmenter>());
    rclcpp::shutdown();
    return 0;
}