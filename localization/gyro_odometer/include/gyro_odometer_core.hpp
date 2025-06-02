#ifndef GYRO_ODOMETER_CORE_HPP_
#define GYRO_ODOMETER_CORE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <tf2_ros/transform_broadcaster.h> 
#include <tf2/convert.h>


#include <tf2/utils.h>  

#include <deque>
#include <memory>
#include <string>

namespace gyro_odometer
{

class GyroOdometerNode : public rclcpp::Node
{
public:
  explicit GyroOdometerNode(const rclcpp::NodeOptions & node_options);

private:
  void callback_vehicle_twist(
    const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr vehicle_twist_msg_ptr);
  void callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr);
  void concat_gyro_and_odometer();
  void timer_callback();
  void publish_data(const geometry_msgs::msg::TwistWithCovarianceStamped & twist_with_cov_raw);

  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    vehicle_twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_raw_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    twist_with_covariance_raw_pub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    twist_with_covariance_pub_;

  std::shared_ptr<tf2_ros::TransformListener> transform_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<diagnostic_updater::Updater> diagnostics_;
  
  geometry_msgs::msg::PoseStamped current_pose_;  // Tracks odom→base_link pose
  rclcpp::Time last_map_to_odom_stamp_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;  

  std::string output_frame_;
  double message_timeout_sec_;

  bool vehicle_twist_arrived_;
  bool imu_arrived_;
  rclcpp::Time latest_vehicle_twist_ros_time_;
  rclcpp::Time latest_imu_ros_time_;
  std::deque<geometry_msgs::msg::TwistWithCovarianceStamped> vehicle_twist_queue_;
  std::deque<sensor_msgs::msg::Imu> gyro_queue_;
  double publish_rate_;  // Add this line
  rclcpp::TimerBase::SharedPtr publish_timer_;  // Add this line
};

}  // namespace gyro_odometer

#endif  // GYRO_ODOMETER_CORE_HPP_