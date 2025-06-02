from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Paths to configuration files
    navigation_pkg_path = get_package_share_directory('wipod_nav')
    global_costmap_path = os.path.join(navigation_pkg_path, 'config', 'global_costmap.yaml')
    local_costmap_path = os.path.join(navigation_pkg_path, 'config', 'local_costmap.yaml')
    teb_params_path = os.path.join(navigation_pkg_path, 'config', 'teb_local_planner_params.yaml')
    smac_params_path = os.path.join(navigation_pkg_path, 'config', 'smac_planner_params.yaml')
    bt_path = os.path.join(navigation_pkg_path, 'config', 'bt_navigator.yaml')
    behavior_path = os.path.join(navigation_pkg_path, 'config', 'behavior_server.yaml')
    smoother_path = os.path.join(navigation_pkg_path, 'config', 'smoother_server.yaml')
    velocity_smoother_path = os.path.join(navigation_pkg_path, 'config', 'velocity_smoother.yaml')

    return LaunchDescription([
        # Global costmap parameters
        DeclareLaunchArgument(
            'global_costmap_param_file',
            default_value=global_costmap_path,
            description='Path to the global costmap configuration file'
        ),
        
        # Local costmap parameters
        DeclareLaunchArgument(
            'local_costmap_param_file',
            default_value=local_costmap_path,
            description='Path to the local costmap configuration file'
        ),

         DeclareLaunchArgument(
            'teb_param_file',
            default_value=teb_params_path,
            description='Path to the teb planner configuration file'
        ),
         DeclareLaunchArgument(
            'smac_param_file',
            default_value=smac_params_path,
            description='Path to the smac_hybrid configuration file'
        ),
         DeclareLaunchArgument(
            'behavior_param_file',
            default_value=behavior_path,
            description='Path to the behavior-sever configuration file'
        ),
        DeclareLaunchArgument(
            'smoother_param_file',
            default_value=smoother_path,
            description='Path to the smoother-sever configuration file'
        ),
        DeclareLaunchArgument(
            'velocity_smoother_param_file',
            default_value=velocity_smoother_path,
            description='Path to the velocity-smoother-sever configuration file'
        ),





        # Planner server node
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[LaunchConfiguration('global_costmap_param_file'), LaunchConfiguration('smac_param_file')]
        ),
        
        # Controller server node
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[LaunchConfiguration('local_costmap_param_file'), LaunchConfiguration('teb_param_file')] #, LaunchConfiguration('smoother_param_file')]
        ),
        
        # Recoveries server node
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[LaunchConfiguration('behavior_param_file')]
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            output='screen',
            parameters=[LaunchConfiguration('smoother_param_file')]
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[LaunchConfiguration('velocity_smoother_param_file')]
        ),
        
        # Behavior Tree Navigator node
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[bt_path]
        ),

        # Load additional configuration parameters
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'autostart': True,
                'node_names': [ 'planner_server', 'controller_server', 'behavior_server', 'bt_navigator', 'smoother_server', 'velocity_smoother']
            }]
        ),
        # Node(
        #     package='wipod_nav',
        #     executable='waypoint_processor',
        #     name='waypoint_processor',
        #     output='screen',
        #     remappings=[('/current_pose','/ndt_pose')],
        #     parameters=[{
        #         'distance_threshold': 7.0,
        #         'map_origin_lat': 12.923903488321232,
        #         'map_origin_lon': 77.50052742264235
        #     }]
        # ),
        # Node(
        #     package='wipod_nav',
        #     executable='osm_2_waypoint.py',
        #     name='osm_2_waypoint',
        #     output='screen'
        # ),
        # Node(
        #     package='wipod_nav',
        #     executable='sim_goal_publisher.py',
        #     name='sim_goal_publisher',
        #     output='screen'
        # ),
        

    ])
