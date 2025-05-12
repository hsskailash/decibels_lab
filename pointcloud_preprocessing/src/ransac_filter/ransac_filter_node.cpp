#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl/ModelCoefficients.h"
#include "pcl/segmentation/sac_segmentation.h"
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include "filter.hpp"

class RansacGroundFilter : public Filter {
private:
    float distance_threshold;
    int max_iterations;
    float voxel_size;
    int mean_k;
    double std_dev_thresh;
    float z_min, z_max;

public:
    RansacGroundFilter(const rclcpp::NodeOptions& options)
        : Filter("ransac_ground_filter", options) {
        
        declare_parameter("distance_threshold", 0.2);
        get_parameter("distance_threshold", distance_threshold);

        declare_parameter("max_iterations", 100);
        get_parameter("max_iterations", max_iterations);

        declare_parameter("voxel_size", 0.1);
        get_parameter("voxel_size", voxel_size);

        declare_parameter("mean_k", 50);
        get_parameter("mean_k", mean_k);

        declare_parameter("std_dev_thresh", 1.0);
        get_parameter("std_dev_thresh", std_dev_thresh);

        declare_parameter("z_min", -1.0);
        get_parameter("z_min", z_min);
        declare_parameter("z_max", 1.0);
        get_parameter("z_max", z_max);

        if (distance_threshold <= 0 || max_iterations <= 0 || voxel_size <= 0 || mean_k <= 0 || std_dev_thresh <= 0) {
            RCLCPP_ERROR(get_logger(), "Invalid filter parameters. Check configuration.");
            rclcpp::shutdown();
        }

        pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic + "/filtered", rclcpp::SensorDataQoS(),
            std::bind(&RansacGroundFilter::pointcloudCallback, this, std::placeholders::_1));

        filtered_pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic + "/non_ground", rclcpp::SensorDataQoS());
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) override {
        if (!msg) {
            RCLCPP_ERROR(get_logger(), "Received a null PointCloud2 message.");
            return;
        }

        OusterPC::Ptr input_cloud = std::make_shared<OusterPC>();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr buffer_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

        try {
            pcl::fromROSMsg(*msg, *input_cloud);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error converting PointCloud2 to PCL: %s", e.what());
            return;
        }

        if (input_cloud->empty()) {
            RCLCPP_WARN(get_logger(), "Input cloud is empty.");
            return;
        }

        buffer_cloud->resize(input_cloud->size());
        for (size_t i = 0; i < input_cloud->size(); ++i) {
            buffer_cloud->points[i].x = input_cloud->points[i].x;
            buffer_cloud->points[i].y = input_cloud->points[i].y;
            buffer_cloud->points[i].z = input_cloud->points[i].z;
        }

        // Apply Voxel Grid filter
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
        voxel_filter.setInputCloud(buffer_cloud);
        voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
        voxel_filter.filter(*buffer_cloud);

        // Apply Statistical Outlier Removal filter
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(buffer_cloud);
        sor.setMeanK(mean_k);
        sor.setStddevMulThresh(std_dev_thresh);
        sor.filter(*buffer_cloud);

        // Step 3: Apply a Passthrough filter to remove very low points (ground threshold)
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(buffer_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min, z_max); // Adjust height limits as needed
        pass.filter(*buffer_cloud);

        


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_buffer = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        OusterPC::Ptr filtered_cloud = std::make_shared<OusterPC>();

        try {
            applyRansacGroundFilter(*buffer_cloud, *output_buffer);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error during ground filtering: %s", e.what());
            return;
        }

        // filtered_cloud->resize(output_buffer->size());
        // for (size_t i = 0; i < output_buffer->size(); ++i) {
        //     filtered_cloud->points[i].x = output_buffer->points[i].x;
        //     filtered_cloud->points[i].y = output_buffer->points[i].y;
        //     filtered_cloud->points[i].z = output_buffer->points[i].z;
        //     filtered_cloud->points[i].intensity = input_cloud->points[i].intensity;
        //     filtered_cloud->points[i].t = input_cloud->points[i].t;
        //     filtered_cloud->points[i].reflectivity = input_cloud->points[i].reflectivity;
        //     filtered_cloud->points[i].ring = input_cloud->points[i].ring;
        //     filtered_cloud->points[i].ambient = input_cloud->points[i].ambient;
        //     filtered_cloud->points[i].range = input_cloud->points[i].range;
        // }

        sensor_msgs::msg::PointCloud2 output_msg;
        try {
            pcl::toROSMsg(*output_buffer, output_msg);
            std_msgs::msg::Header header;
            header.frame_id = lidarFrame;
            header.stamp = get_clock()->now();
            output_msg.header = header;
            filtered_pointcloud_pub_->publish(output_msg);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error converting filtered cloud: %s", e.what());
        }
    }

    void applyRansacGroundFilter(const pcl::PointCloud<pcl::PointXYZRGB>& input, pcl::PointCloud<pcl::PointXYZRGB>& output ) {
        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold);
        seg.setMaxIterations(max_iterations);
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(input);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*ground_indices, *coefficients);

        if (ground_indices->indices.empty()) {
            RCLCPP_WARN(get_logger(), "RANSAC found no ground plane.");
            return;
        }

        pcl::ExtractIndices<pcl::PointXYZRGB> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(ground_indices);
        extract.setNegative(true);
        extract.filter(output);

        RCLCPP_INFO(get_logger(), "Filtered %lu ground points.", ground_indices->indices.size());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RansacGroundFilter>(rclcpp::NodeOptions());

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
