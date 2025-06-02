#include "gyro_odometer_core.hpp"
#include "twist_with_covariance_stamped.hpp" 
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>

#include <deque>
#include <memory>
#include <string>

namespace gyro_odometer
{

GyroOdometerNode::GyroOdometerNode(const rclcpp::NodeOptions & node_options)
: Node("gyro_odometer", node_options),
  output_frame_(declare_parameter<std::string>("output_frame")),
  message_timeout_sec_(declare_parameter<double>("message_timeout_sec")),
  vehicle_twist_arrived_(false),
  imu_arrived_(false)
{
  // Initialize TF buffer and listener
  RCLCPP_INFO(this->get_logger(), "Node is initializing");
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Initialize diagnostics
  diagnostics_ = std::make_unique<diagnostic_updater::Updater>(this);
  diagnostics_->setHardwareID("none");

  // Subscribers
  vehicle_twist_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "vehicle/twist_with_covariance", rclcpp::QoS{100},
    std::bind(&GyroOdometerNode::callback_vehicle_twist, this, std::placeholders::_1));

  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "imu", rclcpp::QoS{100},
    std::bind(&GyroOdometerNode::callback_imu, this, std::placeholders::_1));

  // Publishers
  twist_raw_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("twist_raw", rclcpp::QoS{10});
  twist_with_covariance_raw_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "twist_with_covariance_raw", rclcpp::QoS{10});

  twist_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("twist", rclcpp::QoS{10});
  twist_with_covariance_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "twist_with_covariance", rclcpp::QoS{10});
  
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this); 
  publish_rate_ = declare_parameter<double>("publish_rate", 30.0);
  // message_timeout_sec_ = declare_parameter<double>("message_timeout_sec");
  publish_timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / publish_rate_),
    std::bind(&GyroOdometerNode::timer_callback, this));
    
}

void GyroOdometerNode::callback_vehicle_twist(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr vehicle_twist_msg_ptr)
{
  vehicle_twist_arrived_ = true;
  latest_vehicle_twist_ros_time_ = vehicle_twist_msg_ptr->header.stamp;
  vehicle_twist_queue_.push_back(*vehicle_twist_msg_ptr);  // Just store, no processing
}

void GyroOdometerNode::callback_imu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr)
{
  imu_arrived_ = true;
  latest_imu_ros_time_ = imu_msg_ptr->header.stamp;
  gyro_queue_.push_back(*imu_msg_ptr);  // Just store, no processing
}

void GyroOdometerNode::timer_callback()
{
  concat_gyro_and_odometer();  // Process at fixed rate
}

void GyroOdometerNode::concat_gyro_and_odometer()
{
  if (!vehicle_twist_arrived_ || !imu_arrived_) {
    return;
    }

  const double vehicle_twist_dt =
    std::abs((this->now() - latest_vehicle_twist_ros_time_).seconds());
  const double imu_dt = std::abs((this->now() - latest_imu_ros_time_).seconds());

  if (vehicle_twist_dt > message_timeout_sec_ || imu_dt > message_timeout_sec_) {
    vehicle_twist_queue_.clear();
    gyro_queue_.clear();
    return;
  }

  const auto now = this->now();
  
  // Prune vehicle twist queue
  while (!vehicle_twist_queue_.empty() && 
        (now - vehicle_twist_queue_.front().header.stamp).seconds() > message_timeout_sec_) {
    vehicle_twist_queue_.pop_front();
  }
  
  // Prune IMU queue
  while (!gyro_queue_.empty() && 
        (now - gyro_queue_.front().header.stamp).seconds() > message_timeout_sec_) {
    gyro_queue_.pop_front();
  }

  if (vehicle_twist_queue_.empty() || gyro_queue_.empty()) {
    return;
  }

  geometry_msgs::msg::TransformStamped tf_imu2base;
  try {
    tf_imu2base = tf_buffer_->lookupTransform(
      output_frame_, gyro_queue_.front().header.frame_id, tf2::TimePointZero);
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
    return;
  }

  

  for (auto & gyro : gyro_queue_) {
    geometry_msgs::msg::Vector3Stamped angular_velocity;
    angular_velocity.header = gyro.header;
    angular_velocity.vector = gyro.angular_velocity;

    geometry_msgs::msg::Vector3Stamped transformed_angular_velocity;
    tf2::doTransform(angular_velocity, transformed_angular_velocity, tf_imu2base);

    gyro.header.frame_id = output_frame_;
    gyro.angular_velocity = transformed_angular_velocity.vector;
  }

  double vx_mean = 0;
  geometry_msgs::msg::Vector3 gyro_mean{};


  for (const auto & vehicle_twist : vehicle_twist_queue_)
   {
    geometry_msgs::msg::TwistWithCovarianceStamped transformed_twist;

    // Check if the twist is already in the target frame (base_link)
    if (vehicle_twist.header.frame_id != output_frame_) {
      try {
        // Lookup transform from vehicle_twist's frame to base_link
        geometry_msgs::msg::TransformStamped tf_vehicle_to_base = tf_buffer_->lookupTransform(
          output_frame_,                     // Target frame (base_link)
          vehicle_twist.header.frame_id,     // Source frame (e.g., odom)
          tf2::TimePointZero                 // Use latest available transform
        );

        // Transform the twist to base_link frame
        tf2::doTransform(vehicle_twist, transformed_twist, tf_vehicle_to_base);
      } catch (const tf2::TransformException &ex) {
        RCLCPP_ERROR(
          this->get_logger(),
          "Failed to transform vehicle twist from %s to %s: %s",
          vehicle_twist.header.frame_id.c_str(),
          output_frame_.c_str(),
          ex.what()
        );
        continue; // Skip this twist if transform fails
      }
    } else {
      // No transform needed; use original twist
      transformed_twist = vehicle_twist;
    }

    // Accumulate transformed linear velocity
    vx_mean += transformed_twist.twist.twist.linear.x;
  }
  vx_mean /= static_cast<double>(vehicle_twist_queue_.size());

  for (const auto & gyro : gyro_queue_) {
    gyro_mean.x += gyro.angular_velocity.x;
    gyro_mean.y += gyro.angular_velocity.y;
    gyro_mean.z += gyro.angular_velocity.z;
  }
  gyro_mean.x /= static_cast<double>(gyro_queue_.size());
  gyro_mean.y /= static_cast<double>(gyro_queue_.size());
  gyro_mean.z /= static_cast<double>(gyro_queue_.size());

  geometry_msgs::msg::TwistWithCovarianceStamped twist_with_cov;
  twist_with_cov.header.stamp = this->now();
  twist_with_cov.header.frame_id = output_frame_;
  twist_with_cov.twist.twist.linear.x = vx_mean;
  twist_with_cov.twist.twist.angular = gyro_mean;

  publish_data(twist_with_cov);

  vehicle_twist_queue_.clear();
  gyro_queue_.clear();

  try {
    geometry_msgs::msg::TransformStamped map_to_odom = tf_buffer_->lookupTransform(
      "map", "odom", tf2::TimePointZero
    );

    // Reset integration if EKF updated map→odom
    if (map_to_odom.header.stamp != last_map_to_odom_stamp_) {
      current_pose_.pose.position.x = 0.0;
      current_pose_.pose.position.y = 0.0;
      current_pose_.pose.orientation = tf2::toMsg(tf2::Quaternion::getIdentity());
      last_map_to_odom_stamp_ = map_to_odom.header.stamp;
    }
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(this->get_logger(), "Could not lookup map→odom: %s", ex.what());
  }

  // Integrate velocity (time since last update)
  const double dt = (this->now() - current_pose_.header.stamp).seconds();
  current_pose_.header.stamp = this->now();

  // Update position (x, y)
  const double yaw = tf2::getYaw(current_pose_.pose.orientation);
  current_pose_.pose.position.x += vx_mean * dt * cos(yaw);
  current_pose_.pose.position.y += vx_mean * dt * sin(yaw);

  // Update orientation (yaw)
  const double delta_yaw = gyro_mean.z * dt;
  tf2::Quaternion delta_q;
  delta_q.setRPY(0, 0, delta_yaw);
  tf2::Quaternion current_q;
  tf2::fromMsg(current_pose_.pose.orientation, current_q);
  current_q = current_q * delta_q;
  current_q.normalize();
  current_pose_.pose.orientation = tf2::toMsg(current_q);

  // Publish odom→base_link transform
  geometry_msgs::msg::TransformStamped odom_to_base;
  odom_to_base.header.stamp = this->now();
  odom_to_base.header.frame_id = "odom";
  odom_to_base.child_frame_id = "base_link";
  odom_to_base.transform.translation.x = current_pose_.pose.position.x;
  odom_to_base.transform.translation.y = current_pose_.pose.position.y;
  odom_to_base.transform.translation.z = 0.0;  // 2D
  odom_to_base.transform.rotation = current_pose_.pose.orientation;

  // RCLCPP_INFO(this->get_logger(), "align_time: %f ms", current_pose_.pose.position.x);
  
  tf_broadcaster_->sendTransform(odom_to_base);

}

void GyroOdometerNode::publish_data(
  const geometry_msgs::msg::TwistWithCovarianceStamped & twist_with_cov_raw)
{
  geometry_msgs::msg::TwistStamped twist_raw;
  twist_raw.header = twist_with_cov_raw.header;
  twist_raw.twist = twist_with_cov_raw.twist.twist;

  twist_raw_pub_->publish(twist_raw);
  twist_with_covariance_raw_pub_->publish(twist_with_cov_raw);

  geometry_msgs::msg::TwistWithCovarianceStamped twist_with_covariance = twist_with_cov_raw;
  geometry_msgs::msg::TwistStamped twist = twist_raw;

  if (
    std::fabs(twist_with_cov_raw.twist.twist.angular.z) < 0.01 &&
    std::fabs(twist_with_cov_raw.twist.twist.linear.x) < 0.01) {
    twist.twist.angular.x = 0.0;
    twist.twist.angular.y = 0.0;
    twist.twist.angular.z = 0.0;
    twist_with_covariance.twist.twist.angular.x = 0.0;
    twist_with_covariance.twist.twist.angular.y = 0.0;
    twist_with_covariance.twist.twist.angular.z = 0.0;
  }

  twist_pub_->publish(twist);
  twist_with_covariance_pub_->publish(twist_with_covariance);
}

}  // namespace gyro_odometer

// Main function to run the node as a standalone executable
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gyro_odometer::GyroOdometerNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}