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

# Define source and destination paths for parameters.yaml
set(PARAMS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/config/parameters.yaml")
set(PARAMS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/config")
set(PARAMS_INSTALL_FILE "${PARAMS_INSTALL_DIR}/parameters.yaml")

# Create a custom command to check if the file exists and copy only if it doesn't
install(CODE "
  if(NOT EXISTS \"${PARAMS_INSTALL_FILE}\")
    message(STATUS \"Installing: ${PARAMS_INSTALL_FILE}\")
    file(COPY \"${PARAMS_SRC}\" DESTINATION \"${PARAMS_INSTALL_DIR}\")
  else()
    message(STATUS \"Skipping: ${PARAMS_INSTALL_FILE} (already exists)\")
  endif()
")

# Install the rest of the config directory (excluding parameters.yaml)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  PATTERN "parameters.yaml" EXCLUDE
)

ament_package()
