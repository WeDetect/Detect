cmake_minimum_required(VERSION 3.5)
project(lidar_solution_bringup)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install parameters.yaml directly (will be symlinked with --symlink-install)
install(
  FILES config/parameters.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

# Install the rest of the config directory (excluding parameters.yaml)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  PATTERN "parameters.yaml" EXCLUDE
)

ament_package()
