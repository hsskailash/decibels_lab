cmake_minimum_required(VERSION 3.5)
project(wipod_nav)

# Find the required packages
find_package(ament_cmake_auto REQUIRED)

#find_package(rclcpp REQUIRED)
#find_package(rclpy REQUIRED)
#find_package(nav2_costmap_2d REQUIRED)
#find_package(nav2_util REQUIRED)
#find_package(nav2_voxel_grid REQUIRED) 
#find_package(nav2_smac_planner REQUIRED)
#find_package(teb_local_planner REQUIRED)
#find_package(nav2_bt_navigator REQUIRED)
#find_package(nav2_smoother REQUIRED)
#find_package(nav2_velocity_smoother REQUIRED)
#find_package(geometry_msgs REQUIRED)
#find_package(nav_msgs REQUIRED)

ament_auto_find_build_dependencies()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

ament_auto_add_executable(waypoint_processor src/waypoint_processor.cpp)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()
install(DIRECTORY launch config
        DESTINATION share/${PROJECT_NAME})
install(
  DIRECTORY
  src/
  DESTINATION lib/${PROJECT_NAME}
)
ament_auto_package(
  INSTALL_TO_SHARE
  launch
)

#target_link_libraries(waypoint_processor )

# Install directories
#install(DIRECTORY launch config
#        DESTINATION share/${PROJECT_NAME})
#install(
#  DIRECTORY
#  src/
#  DESTINATION lib/${PROJECT_NAME}
#)
#
## Finalize package
#ament_package()
