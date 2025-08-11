from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ouster_segmenter',
            executable='segmenter_node',
            name='lidar_segmenter',
            output='screen',
            parameters=[{'use_sim_time': False}],
            remappings=[
                ('/ouster/points', '/os_cloud_node/points'),  # Ouster default topic
            ]
        )
    ])