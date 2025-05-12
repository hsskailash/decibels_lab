cmake_minimum_required(VERSION 3.8)
project(ntrip_client)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nmea_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

include_directories(include)

# Create the ntrip_client library
add_library(${PROJECT_NAME}_lib
  src/ntrip_client.cpp
  src/rtcm_parser.cpp
)

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  nmea_msgs
  mavros_msgs
  diagnostic_msgs
  std_msgs
)

target_link_libraries(${PROJECT_NAME}_lib
  Boost::system
  Boost::thread
)

# Create the node executable
add_executable(ntrip_client_node src/ntrip_client_node.cpp)
target_link_libraries(ntrip_client_node
  ${PROJECT_NAME}_lib
)

ament_target_dependencies(ntrip_client_node
  rclcpp
)

install(TARGETS
  ${PROJECT_NAME}_lib
  ntrip_client_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

ament_package()