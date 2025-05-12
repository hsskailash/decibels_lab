from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ntrip_client',
            executable='ntrip_client_node',
            name='ntrip_client',
            output='screen',
            parameters=[{
                # NTRIP Server Configuration
                # Replace these values with your NTRIP caster information
                'host': '43.240.5.42',  # NTRIP caster hostname or IP
                'port': 2105,          # Default NTRIP port is 2101
                'mountpoint': 'IFKP',  # Your NTRIP mountpoint
                'username': 'hssathyakailash.ec18',     # Your NTRIP username
                'password': 'wirin@123',     # Your NTRIP password

                # NMEA and Update Rate Configuration
                # Adjust these based on your GNSS receiver's capabilities
                'nmea_input_rate': 4.0,    # Input NMEA rate in Hz (default: 4.0)
                'update_rate': 1.0,        # Desired rate for sending GGA messages (Hz)

                # Connection Configuration
                'reconnect_delay': 5.0,    # Delay between reconnection attempts (seconds)
                'max_reconnect_attempts': 0,  # 0 for infinite attempts

                # Debug Configuration
                'debug': False,              # Set to True for detailed debug output
            }],
            # Topic Remapping
            # Adjust these if your NMEA input topic or RTCM output topic are different
            remappings=[
                ('nmea', 'nmea'),  # Input NMEA topic
                ('rtcm', 'rtcm')   # Output RTCM topic
            ]
        )
    ])