from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    config_file = os.path.join(
        get_package_share_directory('lidar_fov_filter'),
        'config',
        'lidar_fov_filter_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='lidar_fov_filter',
            executable='lidar_fov_filter',
            name='lidar_fov_filter',
            parameters=[config_file],
            output='screen'
        )
    ])