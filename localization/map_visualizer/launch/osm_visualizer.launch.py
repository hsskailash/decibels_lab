from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="map_visualizer",
            executable="osm_visualizer",
            name="osm_visualizer",
            output="screen",
            emulate_tty=True,
            parameters=[
                {"map_path": "/home/bunty/Pictures/lanelet2_map.osm"},
                {"enable_inc_path_points": False},
                {"interval": 0.5},
                {"origin_lat": 12.923903488321232},  # Default latitude
                {"origin_lon": 77.50052742264235},   # Default longitude
                {"origin_alt": 0.0}                  # Default altitude
            ]
        ),
        Node(
           package="map_visualizer",
           executable="occupancy_pub",
           name="occupancy_pub",
           output="screen",
           emulate_tty=True,
           parameters=[
               {"resolution": 1.0}  # Set the resolution parameter
           ]
        )
    ])