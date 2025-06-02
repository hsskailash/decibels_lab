#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "TransverseMercatorProjector.hpp"

class WaypointProcessor : public rclcpp::Node {
public:
    WaypointProcessor() : Node("waypoint_processor") {
        // Parameters
        declare_parameter("distance_threshold", 7.0);
        declare_parameter("map_origin_lat", 12.923903488321232);
        declare_parameter("map_origin_lon", 77.50052742264235);
        declare_parameter("map_origin_alt", 0.0);

        // Subscribers
        osm_waypoints_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/osm_waypoints", 10,
            std::bind(&WaypointProcessor::osm_waypoints_cb, this, std::placeholders::_1));
            
        // map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        //     "/map", 10,
        //     std::bind(&WaypointProcessor::map_cb, this, std::placeholders::_1));
            
        current_pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/current_pose", 10,
            std::bind(&WaypointProcessor::current_pose_cb, this, std::placeholders::_1));

        // Publishers
        goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);
        markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints_markers", 10);

        // Initialize projector
        auto origin = lanelet::Origin({
            get_parameter("map_origin_lat").as_double(),
            get_parameter("map_origin_lon").as_double(),
            get_parameter("map_origin_alt").as_double()
        });
        projector_ = std::make_shared<lanelet::projection::TransverseMercatorProjector>(origin);
    }

private:
    void osm_waypoints_cb(const nav_msgs::msg::Path::SharedPtr msg) {
        // Convert all waypoints to local coordinates
        local_waypoints_.clear();
        current_goal_index_ = 0;
        
        for (const auto& pose : msg->poses) {
            auto gps_point = lanelet::GPSPoint{
                pose.pose.position.x,  // Latitude from y position
                pose.pose.position.y,  // Longitude from x position
                0.0
            };
            
            auto projected = projector_->forward(gps_point);
            
            geometry_msgs::msg::PoseStamped local_pose;
            local_pose.pose.position.x = projected.x();
            local_pose.pose.position.y = projected.y();
            local_pose.header = pose.header;
            
            local_waypoints_.push_back(local_pose);
        }
        
        if (!local_waypoints_.empty()) {
            // Send first waypoint immediately
            goal_pub_->publish(local_waypoints_[0]);
            publish_markers();
        }
    }
    // void map_cb(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    //     // Store the latest map data (for future obstacle checks)
    //     current_map_ = *msg;
    // }

    void current_pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_pose_ = *msg;
        
        if (local_waypoints_.empty() || current_goal_index_ >= local_waypoints_.size())
            return;

        // Calculate distance to current goal
        auto& current_goal = local_waypoints_[current_goal_index_];
        double dx = current_pose_.pose.position.x - current_goal.pose.position.x;
        double dy = current_pose_.pose.position.y - current_goal.pose.position.y;
        double distance = std::hypot(dx, dy);

        // For first waypoint, wait for explicit success message
        if (current_goal_index_ == 0) return;

        double threshold = get_parameter("distance_threshold").as_double();
        if (distance < threshold) {
            current_goal_index_++;
            if (current_goal_index_ < local_waypoints_.size()) {
                goal_pub_->publish(local_waypoints_[current_goal_index_]);
                publish_markers();
            }
        }
    }

    void publish_markers() {
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker marker;
        
        marker.header.frame_id = "map";
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = marker.scale.y = marker.scale.z = 0.5;

        for (size_t i = 0; i < local_waypoints_.size(); ++i) {
            marker.id = i;
            marker.pose = local_waypoints_[i].pose;
            marker.color.r = (i == current_goal_index_) ? 0.0 : 1.0;
            marker.color.g = (i == current_goal_index_) ? 1.0 : 0.0;
            marker.color.b = 0.0;
            marker.color.a = 1.0;
            markers.markers.push_back(marker);
        }
        
        markers_pub_->publish(markers);
    }

    // Member variables
    std::vector<geometry_msgs::msg::PoseStamped> local_waypoints_;
    size_t current_goal_index_ = 0;
    geometry_msgs::msg::PoseStamped current_pose_;
    std::shared_ptr<lanelet::projection::TransverseMercatorProjector> projector_;
    
    // ROS interfaces
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr osm_waypoints_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr current_pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WaypointProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}