cmake_minimum_required(VERSION 3.14)
project(gyro_odometer)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(fmt REQUIRED)

# Add library
add_library(${PROJECT_NAME} SHARED
  src/gyro_odometer_core.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  diagnostic_msgs
  diagnostic_updater
  fmt
)

# Add executable for the node
add_executable(gyro_odometer_node src/gyro_odometer_core.cpp)

# Link the executable to the library and dependencies
target_link_libraries(gyro_odometer_node ${PROJECT_NAME})
ament_target_dependencies(gyro_odometer_node
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  diagnostic_msgs
  diagnostic_updater
  fmt
)

# Install targets
install(TARGETS
  ${PROJECT_NAME}
  gyro_odometer_node
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY
  include/
  DESTINATION include
)

# Install launch and config directories
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()