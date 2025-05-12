cmake_minimum_required(VERSION 3.5)
project(pointcloud_preprocessing)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
set(OpenCV_STATIC ON) 
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation kdtree search)


# Include directories for your custom header files
include_directories(
  include
)
include_directories(${PCL_INCLUDE_DIRS})

# Create ROS2 interfaces (if needed)
# rosidl_generate_interfaces(${PROJECT_NAME} "msg/CloudInfo.msg" "srv/SaveMap.srv" DEPENDENCIES std_msgs sensor_msgs)

# Add executables for each of your classes
add_executable(crop_box_filter src/crop_box/crop_box.cpp)
ament_target_dependencies(crop_box_filter
  rclcpp
  rclpy
  sensor_msgs
  pcl_conversions
  pcl_msgs
  OpenCV
  PCL
  tf2
  tf2_ros
  nav_msgs
  visualization_msgs
  tf2_eigen
  tf2_geometry_msgs
  
)

add_executable(ray_ground_classifier src/ray_ground_classifier/ray_ground_classifier_node.cpp)
ament_target_dependencies(ray_ground_classifier
  rclcpp
  rclpy
  sensor_msgs
  pcl_conversions
  pcl_msgs
  OpenCV
  PCL
  tf2
  tf2_ros
  nav_msgs
  visualization_msgs
  tf2_eigen
  tf2_geometry_msgs
  
)

add_executable(ransac_filter src/ransac_filter/ransac_filter_node.cpp)
ament_target_dependencies(ransac_filter   
  rclcpp
  rclpy
  sensor_msgs
  pcl_conversions
  pcl_msgs
  OpenCV
  PCL
  tf2
  tf2_ros
  nav_msgs
  visualization_msgs
  tf2_eigen
  tf2_geometry_msgs
  
)
# Link additional dependencies if necessary
target_link_libraries(crop_box_filter ${PCL_LIBRARIES})
target_link_libraries(ray_ground_classifier ${PCL_LIBRARIES})
target_link_libraries(ransac_filter ${PCL_LIBRARIES})
# Install the executables
#ransac_filter
install(TARGETS crop_box_filter ray_ground_classifier ransac_filter
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
install(
  DIRECTORY param
  DESTINATION share/${PROJECT_NAME}/
)

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Install launch files, config files, and other resources if necessary
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME}/)

ament_export_dependencies(rclcpp rclpy sensor_msgs pcl_conversions pcl_msgs OpenCV PCL)
ament_package()

