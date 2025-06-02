#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import pyroutelib3

class OSMWaypointGenerator(Node):
    def __init__(self):
        super().__init__('osm_waypoint_generator')
        
        # Subscriber for end goal
        self.goal_sub = self.create_subscription(
            PoseStamped, '/end_goal', self.goal_callback, 10)
            
        # Publisher for waypoints
        self.waypoints_pub = self.create_publisher(Path, '/osm_waypoints', 10)
        
        # OSM Map loading
        self.graph = self.load_osm_map()
        
        # State variables
        self.current_pose_sub = None
        self.active_goal = None
        self.current_position = None

    def load_osm_map(self):
        """Load OSM map from PBF file"""
        map_path = "/home/wipod_orin/wipod_ws/maps/map.pbf"
        self.get_logger().info(f"Loading OSM map from {map_path}")
        with open(map_path, "rb") as f:
            return pyroutelib3.osm.Graph.from_file(
                pyroutelib3.osm.CarProfile(), f)

    def goal_callback(self, msg):
        """Handle new goal and activate pose subscription"""
        if self.active_goal:
            self.get_logger().warn("Already processing a goal, ignoring new one")
            return
            
        # Store new goal
        self.active_goal = (
            msg.pose.position.x,  # Latitude
            msg.pose.position.y   # Longitude
        )
        
        # Create temporary pose subscription
        if not self.current_pose_sub:
            self.current_pose_sub = self.create_subscription(
                PoseStamped,
                '/gnss_pose',
                self.pose_callback,
                10
            )
            self.get_logger().info("Subscribed to current pose")

    def pose_callback(self, msg):
        """Process single pose and calculate route"""
        # Get current position
        self.current_position = (
            msg.pose.position.x,  # Latitude
            msg.pose.position.y   # Longitude
        )
        
        # Calculate route
        if self.active_goal:
            route = self.calculate_route(self.current_position, self.active_goal)
            if route:
                self.publish_waypoints(route)
            
            # Cleanup
            self.destroy_subscription(self.current_pose_sub)
            self.current_pose_sub = None
            self.active_goal = None
            self.get_logger().info("Route processed, unsubscribed from pose")

    def calculate_route(self, start_gps, end_gps):
        """Calculate route using pyroutelib3"""
        try:
            start_node = self.graph.find_nearest_node(start_gps)
            end_node = self.graph.find_nearest_node(end_gps)
            
            route = pyroutelib3.find_route(self.graph, start_node.id, end_node.id)
            return [self.graph.get_node(node).position for node in route] if route else None
            
        except Exception as e:
            self.get_logger().error(f"Routing failed: {str(e)}")
            return None

    def publish_waypoints(self, route_points):
        """Convert route to Path message"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for lat, lon in route_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = lat
            pose.pose.position.y = lon
            path_msg.poses.append(pose)
            
        self.waypoints_pub.publish(path_msg)
        self.get_logger().info(f"Published {len(route_points)} waypoints")

def main(args=None):
    rclpy.init(args=args)
    node = OSMWaypointGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # node.get_logger().info("Node shutdown by user request")
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()

if __name__ == '__main__':
    main() 

