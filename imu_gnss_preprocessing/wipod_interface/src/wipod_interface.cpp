
#include <rclcpp/rclcpp.hpp>
#include <can_msgs/msg/frame.hpp>  // Replace with the actual message type
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include "TransverseMercatorProjector.hpp"
#include <tf2_ros/transform_broadcaster.h>

class WipodInterface : public rclcpp::Node {
public:
    WipodInterface() : Node("can_subscriber_node") {
        // Declare a ROS parameter to set the desired CAN ID
        std::uint32_t steer_can_id_;
        std::uint32_t kelly_left_id_;
        std::uint32_t kelly_right_id_;
        std::uint32_t can_send_id_;
        std::uint32_t gear_can_id_;
        std::string base_frame_id_;
        std::wint_t min_steer_;
        std::wint_t max_steer_;
        std::float_t left_steer_;
        std::float_t right_steer_;
        std::double_t wheel_base_;
        std::double_t tire_radius_;
        std::float_t speed_min_;

        std::double_t origin_lat_;
        std::double_t origin_lon_;
        std::double_t origin_alt_;
        
        steer_can_id_= declare_parameter("steer_can_id",403105078); 
        kelly_left_id_= declare_parameter("kelly_left_id",217128454);
        kelly_right_id_= declare_parameter("kelly_right_id",217128453);
        gear_can_id_= declare_parameter("gear_can_id",217128709);
        can_send_id_= declare_parameter("can_send_id",403039337);
        base_frame_id_ = declare_parameter("base_frame_id", "base_link");
        min_steer_ = declare_parameter("min_steer",188);
        max_steer_ = declare_parameter("max_steer",916);
        left_steer_ = declare_parameter("left_steer", 0.523598776);
        right_steer_ = declare_parameter("right_steer", -0.523598776);
        tire_radius_ = declare_parameter("tire_radius", 0.334);
        wheel_base_ = declare_parameter("wheel_base", 2.75);
        speed_min_ = declare_parameter("speed_min",31.0);

        origin_lat_ = declare_parameter("origin_lat", 12.923903488321232); // Default latitude
        origin_lon_ = declare_parameter("origin_lon", 77.50052742264235);  // Default longitude
        origin_alt_ = declare_parameter("origin_alt", 0.0);  // Default longitude


        // Get the desired CAN ID from the ROS parameter server
        this->get_parameter("steer_can_id", steer_can_id);
        this->get_parameter("kelly_left_id", kelly_left_id);
        this->get_parameter("gear_can_id", gear_can_id);
        this->get_parameter("can_send_id", can_send_id);
        this->get_parameter("base_frame_id", base_frame_id);
        this->get_parameter("kelly_right_id", kelly_right_id);
        this->get_parameter("max_steer", max_steer);
        this->get_parameter("min_steer", min_steer);
        this->get_parameter("left_steer", left_steer);
        this->get_parameter("right_steer", right_steer);
        this->get_parameter("tire_radius", tire_radius);
        this->get_parameter("wheel_base", wheel_base);
        this->get_parameter("speed_min", speed_min);
        this->get_parameter("origin_lat", origin_lat);
        this->get_parameter("origin_lon", origin_lon);
        this->get_parameter("origin_alt", origin_alt);

        lanelet::Origin origin({origin_lat, origin_lon, origin_alt});
        projector_ = std::make_shared<lanelet::projection::TransverseMercatorProjector>(origin);
        // Subscribe to the CAN topic
        can_subscription_ = this->create_subscription<can_msgs::msg::Frame>("/from_can_bus",10, std::bind(&WipodInterface::canCallback, this, std::placeholders::_1));
        //Subsrcibe to gnss
        pos_sub_ = this->create_subscription<sensor_msgs::msg::NavSatFix>("/gnss",10, std::bind(&WipodInterface::positionCallback, this, std::placeholders::_1));
        quat_sub_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>("/filter/quaternion",10, std::bind(&WipodInterface::quatCallback, this, std::placeholders::_1));
        //Subscribe to command topic
        control_command_ = this->create_subscription<geometry_msgs::msg::Twist>("/cmd_vel", 10, std::bind(&WipodInterface::callbackControlCmd, this, std::placeholders::_1));
        // Publisher to Nav2
        odom_pub_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/vehicle/twist_with_covariance_stamped", rclcpp::QoS{100});
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/pose_with_covariance_stamped", rclcpp::QoS{100});
        // Publisher from Nav2
        to_can_ = create_publisher<can_msgs::msg::Frame>("/to_can_bus", rclcpp::QoS{1});

        odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odometry_pub", rclcpp::QoS{100});
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    }

private:
        
    rclcpp::Publisher<can_msgs::msg::Frame>::SharedPtr to_can_;
    // rclcpp::Service<ControlModeCommand>::SharedPtr control_mode_server_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Subscription<can_msgs::msg::Frame>::SharedPtr can_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr pos_sub_;
    rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr quat_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr control_command_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    std::shared_ptr<lanelet::projection::TransverseMercatorProjector> projector_;

    uint32_t steer_can_id; 
    uint32_t kelly_left_id;
    uint32_t kelly_right_id;
    uint32_t gear_can_id;
    uint32_t can_send_id;
    uint8_t control_mode;
    uint8_t gear_status;
    uint8_t gear_value;

    std::string base_frame_id;
    wint_t min_steer;
    wint_t max_steer;
    float_t left_steer;
    float_t right_steer;
    double_t wheel_base;
    double_t tire_radius;
    wint_t steer;
    float_t steer_rad;
    wint_t rpm;
    float_t speed_ms;
    float_t speed_mps;
    float_t steer_cmd;
    float_t steering_tire_rotation_rate;
    float_t speed_cmd;
    float_t speed_cmad;
    float_t acceleration;
    float_t speed_min;
    float_t steer_rate;
    wint_t gear_cmd;
    wint_t steer_can;
    wint_t speed_can;
    wint_t accel_can;
    wint_t decel_can;
    wint_t speed_can1;
    uint8_t brake_can;
    uint8_t gear_can;
    uint8_t steer_0;
    uint8_t steer_1;
    uint8_t speed_2;
    uint8_t speed_3;

    float_t pos_x;
    float_t pos_y;
    float_t pos_z;

    float_t cov_x;
    float_t cov_y;
    float_t cov_z;

    float_t quat_x;
    float_t quat_y;
    float_t quat_z;
    float_t quat_w;
    
    double_t origin_lat;
    double_t origin_lon;
    double_t origin_alt;
    

    



    void positionCallback(const sensor_msgs::msg::NavSatFix::SharedPtr msg){

        pos_x = msg->latitude;
        pos_y = msg->longitude;
        pos_z = msg->altitude;

        lanelet::GPSPoint gps_point{pos_x, pos_y, pos_z};
        lanelet::BasicPoint3d projected = projector_->forward(gps_point);

        pos_x = projected.x();
        pos_y = projected.y();
        // pos_z = projected.z();
        pos_z = 0.0;

        cov_x = msg->position_covariance[0];
        cov_y = msg->position_covariance[4];
        cov_z = msg->position_covariance[8];

        std_msgs::msg::Header header;
        header.frame_id = "map";
        header.stamp = get_clock()->now(); 

        {
            geometry_msgs::msg::PoseWithCovarianceStamped pose_feedback;
            pose_feedback.header = header;
            // pose_feedback.child_frame_id = "chassis_link";
            pose_feedback.pose.pose.position.x = pos_x;
            pose_feedback.pose.pose.position.y = pos_y;
            pose_feedback.pose.pose.position.z = pos_z;

            pose_feedback.pose.pose.orientation.x = quat_x;
            pose_feedback.pose.pose.orientation.y = quat_y;
            pose_feedback.pose.pose.orientation.z = quat_z;
            pose_feedback.pose.pose.orientation.w = quat_w;

            pose_feedback.pose.covariance[0] = cov_x;
            pose_feedback.pose.covariance[7] = cov_y;
            pose_feedback.pose.covariance[14] = cov_z;

            
            pose_pub_->publish(pose_feedback);
        }

        {
            geometry_msgs::msg::TransformStamped ndt_tf;
            ndt_tf.header.stamp = this->get_clock()->now();
            ndt_tf.header.frame_id = "map";  // Parent frame
            ndt_tf.child_frame_id = base_frame_id;     // Typically "base_link"

            // Use pos_x and pos_y as the translation; set z to 0 (or pos_z if needed)
            ndt_tf.transform.translation.x = pos_x;
            ndt_tf.transform.translation.y = pos_y;
            ndt_tf.transform.translation.z = 0.0;  // Adjust as needed

            // Use the orientation values received from quatCallback.
            ndt_tf.transform.rotation.x = quat_x;
            ndt_tf.transform.rotation.y = quat_y;
            ndt_tf.transform.rotation.z = quat_z;
            ndt_tf.transform.rotation.w = quat_w;

             // Publish the transform
            // tf_broadcaster_->sendTransform(ndt_tf);
        }

    }

    void quatCallback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg){

        quat_x = msg->quaternion.x;
        quat_y = msg->quaternion.y;
        quat_z = msg->quaternion.z;
        quat_w = msg->quaternion.w;
    }


    void canCallback(const can_msgs::msg::Frame::SharedPtr msg) {
        // Check if the received message's CAN ID matches the desired CAN ID
        std_msgs::msg::Header header;
        header.frame_id = base_frame_id;
        header.stamp = get_clock()->now(); 
        if (msg->id == steer_can_id) {
            // Extract data from the CAN message
            uint32_t can_id = msg->id;           // Example: Extract CAN ID
            const auto& array_data = msg->data;
            bool is_extended = msg->is_extended;
            bool is_error = msg->is_error;
            uint8_t len = msg->dlc;

            std::vector<uint8_t> data(array_data.begin(), array_data.end());   // Example: Extract data as a byte array
            

            steer = (array_data[1] << 8) + array_data[0];
            steer_rad = left_steer + ((right_steer-left_steer)/(max_steer-min_steer))*(steer-min_steer);
        }
        

                       
        
        if (msg->id == gear_can_id)
        {
            const auto& gear_data = msg->data;
            std::vector<uint8_t> data(gear_data.begin(), gear_data.end());
            gear_value = gear_data[4];
        }

        if (msg->id == kelly_left_id)
        {
            const auto& array_data = msg->data;
            std::vector<uint8_t> data(array_data.begin(), array_data.end());
            
            rpm = (array_data[1] << 8) + array_data[0];  
            speed_ms = (2.0*3.14159265359*tire_radius)*(rpm/60.0);
           
        }

        if (gear_value == 1 || gear_value == 5)
        {
            speed_mps = speed_ms;
        }   
        else if (gear_value == 2 || gear_value == 10)
        {
            speed_mps = -1*speed_ms;
        } 

        {
            geometry_msgs::msg::TwistWithCovarianceStamped odom_feedback;
            odom_feedback.header = header;
            // odom_feedback.child_frame_id = "chassis_link";
            // odom_feedback.pose.pose.position.x = pos_x;
            // odom_feedback.pose.pose.position.y = pos_y;
            // odom_feedback.pose.pose.position.z = pos_z;

            // odom_feedback.pose.pose.orientation.x = quat_x;
            // odom_feedback.pose.pose.orientation.y = quat_y;
            // odom_feedback.pose.pose.orientation.z = quat_z;
            // odom_feedback.pose.pose.orientation.w = quat_w;

            // odom_feedback.pose.covariance[0] = cov_x;
            // odom_feedback.pose.covariance[7] = cov_y;
            // odom_feedback.pose.covariance[14] = cov_z;

            odom_feedback.twist.twist.linear.x = speed_mps;
            
            odom_feedback.twist.twist.angular.z = steer_rad;
            odom_pub_->publish(odom_feedback);
        }

        {
            nav_msgs::msg::Odometry odometry_feedback;
            odometry_feedback.header = header;
            odometry_feedback.child_frame_id = "chassis_link";
            odometry_feedback.pose.pose.position.x = pos_x;
            odometry_feedback.pose.pose.position.y = pos_y;
            odometry_feedback.pose.pose.position.z = pos_z;

            odometry_feedback.pose.pose.orientation.x = quat_x;
            odometry_feedback.pose.pose.orientation.y = quat_y;
            odometry_feedback.pose.pose.orientation.z = quat_z;
            odometry_feedback.pose.pose.orientation.w = quat_w;

            odometry_feedback.pose.covariance[0] = cov_x;
            odometry_feedback.pose.covariance[7] = cov_y;
            odometry_feedback.pose.covariance[14] = cov_z;

            odometry_feedback.twist.twist.linear.x = speed_mps;
            
            odometry_feedback.twist.twist.angular.z = steer_rad;
            odometry_pub_->publish(odometry_feedback);
        }

        


    }
    std::array<uint8_t,8>data_2_bus;
    
    void callbackControlCmd(const geometry_msgs::msg::Twist::ConstSharedPtr msg)
    {
        steer_cmd = msg->angular.z;

        
        // steer_rate = msg->lateral.steering_tire_rotation_rate;
        speed_cmad = msg->linear.x;
        // acceleration = msg->longitudinal.acceleration;

       

        steer_can = ((steer_cmd - left_steer) / (right_steer - left_steer)) * (max_steer - min_steer) + min_steer;

        if (speed_cmad < 0 )
        {
            speed_cmd = -speed_cmad;

            if (speed_cmad < 2.0 && speed_cmad != 0.0)
            {
                speed_cmad = 2.0;
            }
            else{ speed_cmad = speed_cmad;}
            gear_can = 1;
        }
        else{
            speed_cmd = speed_cmad;


            if (speed_cmad < 2.0 && speed_cmad != 0.0)
            {
                speed_cmad = 2.0;
            }
            else{ speed_cmad = speed_cmad;}

            gear_can = 0;
            }

        speed_can = ((speed_cmad*60)/(2.0*M_PI*tire_radius));

        // if (speed_cmd > 0.0 && speed_cmd < 1.1)
        // {
        //     speed_can1 = speed_min;
        // }
        // else if (speed_cmd > 1.1)
        // {
        //     speed_can1 = speed_can;
        // }

        // if (acceleration < 0)
        // {
        //     accel_can = -acceleration*10;
        //     decel_can = 2;
        // }
        // else if (acceleration > 0)
        // {
        //     accel_can = acceleration*10;
        //     decel_can = 1;
        // }

        // if(acceleration>-0.5 && acceleration<0.0005)
        //  {brake_can = 1;}
        // else if(acceleration < -0.5 )
        //  {brake_can = 2;}
        // else if(acceleration>0.0)
        //  {brake_can = 0;} 

        steer_0 = steer_can & 0xFF;
        data_2_bus[0] = steer_0;
        steer_1 = (steer_can>>8) & 0xFF;
        data_2_bus[1] = steer_1;
        speed_2 = speed_can & 0xFF;
        data_2_bus[2] = speed_2;
        speed_3 = (speed_can>>8) & 0xFF;   
        data_2_bus[3] = speed_3;
        data_2_bus[4] = gear_can;
        // data_2_bus[5] = brake_can;
        // data_2_bus[6] = decel_can;
        // data_2_bus[7] = accel_can; 

        {

           std_msgs::msg::Header header;
           header.frame_id = base_frame_id;
           header.stamp = get_clock()->now();     
         //    std::array<uint8_t,8>data_2_bus = {steer_0, steer_1, speed_2, speed_3, gear_can, brake_can, acceleration, 0};
           std::vector<uint8_t> car_dat(data_2_bus.begin(),data_2_bus.end());
           bool is_extended = false;
           bool is_error = false;
           can_msgs::msg::Frame can_pub;
           can_pub.header = header;
           can_pub.id = can_send_id;
           can_pub.data = data_2_bus;
           can_pub.dlc = 8;
           can_pub.is_extended = is_extended;
           can_pub.is_error = is_error;
           to_can_->publish(can_pub);
    
        //    RCLCPP_INFO(this->get_logger(), "Received CAN ID: %u", can_send_id);
        //    RCLCPP_INFO(this->get_logger(), "Data: ");
            // for (const auto& byte : data_2_bus) {
                // RCLCPP_INFO(this->get_logger(), "%d ", byte);
            // }
       
        }

        RCLCPP_INFO(this->get_logger(), "Received speed: %d", speed_can);
        RCLCPP_INFO(this->get_logger(), "Received steer: %d", steer_can);
        RCLCPP_INFO(this->get_logger(), "Received brake: %d", brake_can); 
        RCLCPP_INFO(this->get_logger(), "Received acceleration: %d", accel_can); 

        //  RCLCPP_INFO(this->get_logger(), "Received steer0: %d", steer_0); 
        //  RCLCPP_INFO(this->get_logger(), "Received steer1: %d", steer_1); 
        //  RCLCPP_INFO(this->get_logger(), "Received speed2: %d", speed_2); 
        //  RCLCPP_INFO(this->get_logger(), "Received speed3: %d", speed_3); 
        
 
    }

    // void callbackGearCmd(const autoware_auto_vehicle_msgs::msg::GearCommand::ConstSharedPtr msg) 
    // {
    //     gear_cmd = msg->command;

    //     if(gear_cmd == 2)
    //     {gear_can = 0;}
    //     else if(gear_cmd == 20)
    //     {gear_can = 1;}
    //     else 
    //     {gear_can = 5;}
    //     data_2_bus[4] = gear_can;

    //     {

    //        std_msgs::msg::Header header;
    //        header.frame_id = base_frame_id;
    //        header.stamp = get_clock()->now();     
    //      //    std::array<uint8_t,8>data_2_bus = {steer_0, steer_1, speed_2, speed_3, gear_can, brake_can, acceleration, 0};
    //        std::vector<uint8_t> car_dat(data_2_bus.begin(),data_2_bus.end());
    //        bool is_extended = true;
    //        bool is_error = false;
    //        can_msgs::msg::Frame can_pub;
    //        can_pub.header = header;
    //        can_pub.id = can_send_id;
    //        can_pub.data = data_2_bus;
    //        can_pub.dlc = 8;
    //        can_pub.is_extended = is_extended;
    //        can_pub.is_error = is_error;
    //        to_can_->publish(can_pub);
    
    //     //    RCLCPP_INFO(this->get_logger(), "Received CAN ID: %u", can_send_id);
    //     //    RCLCPP_INFO(this->get_logger(), "Data: ");
    //         // for (const auto& byte : data_2_bus) {
    //             // RCLCPP_INFO(this->get_logger(), "%d ", byte);
    //         // }
       
    //     }
    // }

    // void canPublishCallback(const can_msgs::msg::Frame::SharedPtr msg)
   
    // Other private members and functions

    
};


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WipodInterface>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


