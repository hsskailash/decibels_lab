#!/usr/bin/env python3
import rclpy
import json
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        
        # Initialize variables
        self.json_file = '/home/wipod_orin/wipod_ws/src/wipod_ws/planner/wipod_nav/config/goals.json'
        self.current_goal = None
        self.last_published_goal = None
        self.send_goal = None
        
        # Create publisher and timer
        self.goal_pub = self.create_publisher(PoseStamped, '/end_goal', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("Pose Publisher Node Started!")

    def load_goal_from_json(self):
        """Load and validate goal from JSON file"""
        try:
            if os.path.exists(self.json_file):
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                    self.send_goal = data["goal_point"]["send_goal"]
                    new_lat = data["goal_point"]["latitude"]
                    new_lon = data["goal_point"]["longitude"]
                    
                    # Create a tuple for comparison
                    new_goal = (new_lat, new_lon)                
                    if(self.send_goal == 1):
                        if (new_goal != self.current_goal):
                            self.current_goal = new_goal
                            self.get_logger().info("New goal detected in JSON file")
                            return True
                    else: 
                        return False
            return False
        except Exception as e:
            self.get_logger().error(f"Error reading JSON: {str(e)}")
            return False

    def timer_callback(self):
        if self.load_goal_from_json():
            if(self.send_goal == 1):
            # Only publish if we have a new valid goal
                msg = PoseStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'map'
                msg.pose.position.x = self.current_goal[0]
                msg.pose.position.y = self.current_goal[1]
                msg.pose.position.z = 0.0
                msg.pose.orientation.w = 1.0

                self.goal_pub.publish(msg)
                self.last_published_goal = self.current_goal
                self.get_logger().info(f"Published new goal: {self.current_goal}")
            else:
                return

def main(args=None):
    rclpy.init(args=args)
    pose_publisher = None
    try:
        pose_publisher = PosePublisher()
        rclpy.spin(pose_publisher)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if pose_publisher is not None:
            pose_publisher.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()

if __name__ == '__main__':
    main()
