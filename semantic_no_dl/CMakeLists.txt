cmake_minimum_required(VERSION 3.8)
project(ouster_segmenter)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation features)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(segmenter_node
  src/segmenter_node.cpp
)
ament_target_dependencies(segmenter_node
 rclcpp
 visualization_msgs
 sensor_msgs
 pcl_conversions
 pcl_msgs
 PCL
)

target_link_libraries(segmenter_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS segmenter_node
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch params
  DESTINATION share/${PROJECT_NAME}
)

ament_package()