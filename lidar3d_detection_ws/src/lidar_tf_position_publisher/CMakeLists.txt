cmake_minimum_required(VERSION 3.5)
project(lidar_tf_position_publisher)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Find ament packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
  include
  ${YAML_CPP_INCLUDE_DIRS}
)

# Create executable
add_executable(lidar_tf_position_publisher_node src/lidar_tf_position_publisher_node.cpp)

# Link dependencies
ament_target_dependencies(lidar_tf_position_publisher_node
  rclcpp
  visualization_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
)

# Link target libraries
target_link_libraries(lidar_tf_position_publisher_node 
  yaml-cpp 
  ${tf2_LIBRARIES}
  ${tf2_geometry_msgs_LIBRARIES}
  ${tf2_ros_LIBRARIES}

)

# Install targets to the install/lib/${PROJECT_NAME} directory
install(TARGETS
  lidar_tf_position_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)


# Install the YAML config file to the install/share/lidar_tf_position_publisher/config/
install(
  FILES config/config.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

#copy the params folder to the install directory to the install/share/lidar_tf_position_publisher/params/
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params
)

ament_package()
