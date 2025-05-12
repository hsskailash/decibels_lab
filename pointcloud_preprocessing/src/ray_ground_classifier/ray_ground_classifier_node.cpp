#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "filter.hpp"

class RayGroundFilter : public Filter {
private:
    float min_radius, max_radius, ground_slope, height_threshold, radial_divider_angle;
    std::vector<float> ring_height_thresholds;  // New: Height thresholds per ring

public:
    RayGroundFilter(const rclcpp::NodeOptions& options)
        : Filter("ray_ground_filter", options) {
        declare_parameter("min_radius", 0.0);
        get_parameter("min_radius", min_radius);

        declare_parameter("max_radius", 50.0);
        get_parameter("max_radius", max_radius);

        declare_parameter("ground_slope", 0.1);
        get_parameter("ground_slope", ground_slope);

        declare_parameter("height_threshold", 0.2);
        get_parameter("height_threshold", height_threshold);

        declare_parameter("radial_divider_angle", 0.2);
        get_parameter("radial_divider_angle", radial_divider_angle);

        // Define height thresholds per ring (adjust as needed)
        ring_height_thresholds = {0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7}; // Lower for first rings

        if (min_radius < 0 || max_radius <= min_radius || ground_slope <= 0 || height_threshold <= 0 || radial_divider_angle <= 0) {
            RCLCPP_ERROR(get_logger(), "Invalid parameters. Check configuration.");
            rclcpp::shutdown();
        }

        pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic , rclcpp::SensorDataQoS(),
            std::bind(&RayGroundFilter::pointcloudCallback, this, std::placeholders::_1));

        filtered_pointcloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic + "/ground_filtered", rclcpp::SensorDataQoS());
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) override {
        if (!msg) {
            RCLCPP_ERROR(get_logger(), "Received a null PointCloud2 message.");
            return;
        }

        auto cloud = std::make_shared<OusterPC>();
        try {
            pcl::fromROSMsg(*msg, *cloud);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error converting PointCloud2 to PCL: %s", e.what());
            return;
        }

        if (cloud->empty()) {
            RCLCPP_WARN(get_logger(), "Input cloud is empty.");
            return;
        }

        input_cloud_ = cloud;
        auto filtered_cloud = std::make_shared<OusterPC>();
        output_cloud_ = filtered_cloud;

        try {
            applyRayGroundFilter(*cloud, *filtered_cloud);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error during ground filtering: %s", e.what());
            return;
        }

        if (filtered_cloud->empty()) {
            RCLCPP_WARN(get_logger(), "Filtered cloud is empty.");
        } else {
            sensor_msgs::msg::PointCloud2 output_msg;
            try {
                pcl::toROSMsg(*filtered_cloud, output_msg);
                std_msgs::msg::Header header;
                header.frame_id = lidarFrame;
                header.stamp = get_clock()->now();
                output_msg.header = header;
                filtered_pointcloud_pub_->publish(output_msg);
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "Error converting filtered cloud: %s", e.what());
            }
        }
    }

    void applyRayGroundFilter(const OusterPC& input, OusterPC& output) {
        std::map<int, OusterPC> radial_bins;
    
        // Resize ring_height_thresholds to match the number of rings in your LiDAR
        const int num_rings = 32;  // Adjust this based on your LiDAR
        ring_height_thresholds.resize(num_rings, height_threshold);  // Default to height_threshold
    
        // Set stricter thresholds for the first 5 rings
        for (int i = 0; i < 5; ++i) {
            ring_height_thresholds[i] = 0.05f;  // Stricter threshold for the first 5 rings
        }
    
        for (const auto& point : input.points) {
            float range = std::sqrt(point.x * point.x + point.y * point.y);
            if (range < min_radius || range > max_radius) {
                continue;
            }
    
            float angle = std::atan2(point.y, point.x);
            int bin_index = static_cast<int>(angle / radial_divider_angle);
    
            radial_bins[bin_index].push_back(point);
        }
    
        if (radial_bins.empty()) {
            RCLCPP_WARN(get_logger(), "No valid points after radial binning.");
            return;
        }
    
        for (auto& [bin_idx, bin_points] : radial_bins) {
            if (bin_points.empty()) {
                continue;
            }
    
            std::sort(bin_points.points.begin(), bin_points.points.end(),
                      [](const ouster_ros::Point& a, const ouster_ros::Point& b) {
                          return std::sqrt(a.x * a.x + a.y * a.y) <
                                 std::sqrt(b.x * b.x + b.y * b.y);
                      });
    
            // Initialize with the first point (assumed to be ground)
            float last_ground_z = bin_points.points.front().z;
            float last_ground_range = std::sqrt(
                bin_points.points.front().x * bin_points.points.front().x +
                bin_points.points.front().y * bin_points.points.front().y
            );
    
            for (const auto& point : bin_points.points) {
                float range = std::sqrt(point.x * point.x + point.y * point.y);
                float height_diff = point.z - last_ground_z;
                float slope = std::abs(height_diff) / (range - last_ground_range);
    
                // Extract ring number
                int ring = point.ring;
    
                // Ensure the ring index is within bounds
                if (ring < 0 || ring >= ring_height_thresholds.size()) {
                    RCLCPP_WARN(get_logger(), "Invalid ring index: %d. Using default height threshold.", ring);
                    continue;
                }
    
                // Adaptive height threshold for the ring
                float adaptive_height_threshold = ring_height_thresholds[ring];
    
                if (slope <= ground_slope && std::abs(height_diff) <= adaptive_height_threshold) {
                    // Ground point: update last_ground_z and last_ground_range
                    last_ground_z = point.z;
                    last_ground_range = range;
                } else {
                    output.push_back(point);
                }
            }
        }
    
        if (output.empty()) {
            RCLCPP_WARN(get_logger(), "All points classified as ground.");
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RayGroundFilter>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}
