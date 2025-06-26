// #include <rclcpp/rclcpp.hpp>
// #include <can_msgs/msg/frame.hpp>
// #include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp>
// #include <autoware_auto_vehicle_msgs/msg/steering_report.hpp>


// class WipodInterface : public rclcpp::Node 
// {
// public:
// //   using ActuationCommandStamped = tier4_vehicle_msgs::msg::ActuationCommandStamped;
// //   using ActuationStatusStamped = tier4_vehicle_msgs::msg::ActuationStatusStamped;
// //   using SteeringWheelStatusStamped = tier4_vehicle_msgs::msg::SteeringWheelStatusStamped;
// //   using ControlModeCommand = autoware_auto_vehicle_msgs::srv::ControlModeCommand;
//   WipodInterface();
// private:
//     rclcpp::Publisher<can_msgs::msg::Frame>::SharedPtr can_pub_;
//     rclcpp::Subscription<can_msgs::msg::Frame>::SharedPtr can_subscription_;
//     rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::VelocityReport>::SharedPtr vehicle_twist_pub_;
//     rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::SteeringReport>::SharedPtr steering_status_pub_;

//     void canCallback(const can_msgs::msg::Frame::SharedPtr msg);

//     uint32_t desired_can_id;  
//     const auto& array_data;
//     uint32_t can_id;
//     bool is_extended;
//     bool is_error;
//     uint8_t len;
//     std::vector<uint8_t> data(array_data.begin(), array_data.end());

//     std::string base_frame_id_;


// }