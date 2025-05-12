import launch
from launch_ros.actions import Node

def generate_launch_description():
    return launch.LaunchDescription([
        Node(
            package='ntrip',
            executable='ntrip',
            name='ntrip_client',
            output='screen',
            parameters=[
                {'ip': '43.240.5.42'},  # Change to the IP address of Your NTRIP service
                {'port': 2105},  # Change to your port number, WGS84
                {'user': 'hssathyakailash.ec18'},  # Change to your username
                {'passwd': 'wirin@123'},  # Change to your password
                {'mountpoint': 'IFKP'},  # Change to your mountpoint
                {'report_interval': 1} # the report interval to the NTRIP Caster, default is 1 sec
            ]
        ),
    ])
    
