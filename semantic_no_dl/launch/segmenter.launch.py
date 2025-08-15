from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ouster_segmenter',
            executable='segmenter_node',
            name='lidar_segmenter',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'z_min': -5.0,     # Add Z-range parameters
                'z_max': 20.0,
                # Ouster-specific tuning
                'voxel_size': 0.15,
                'min_cluster_size': 15,
                'cluster_tolerance': 0.5
            }],
            remappings=[
                ('/ouster/points', '/ouster/points/filtered'),  # Uncomment for Ouster
            ]
        )
    ])