cmake_minimum_required(VERSION 3.5)
project(dxl_driver)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(dynamixel_sdk REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

# Add service
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetHomePosition.srv"
  "srv/SetTorqueEnable.srv"
  "srv/SetMaxVelocity.srv"
)
include_directories(include)

# Build
add_executable(dxl_driver_node src/dxl_driver.cpp)
target_include_directories(dxl_driver_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/dxl_driver>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(dxl_driver_node
  rclcpp
  dynamixel_sdk
  sensor_msgs
)

# Install
install(TARGETS
  dxl_driver_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch 
  config
  DESTINATION share/${PROJECT_NAME}/
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(dxl_driver_node
  ${cpp_typesupport_target})

ament_export_dependencies(rosidl_default_runtime)

ament_package()
