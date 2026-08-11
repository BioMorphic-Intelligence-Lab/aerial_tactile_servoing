cmake_minimum_required(VERSION 3.8)
project(px4_uam_sim)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
#find_package(Eigen)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
set(Eigen3_DIR /usr/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)

include_directories(/opt/ros/jazzy/include /usr/include/eigen3)

add_executable(uam_handler src/uam_handler.cpp)
ament_target_dependencies(uam_handler rclcpp px4_msgs geometry_msgs std_msgs tf2 nav_msgs)

add_executable(mocap_forwarder src/mocap_forwarder.cpp)
ament_target_dependencies(mocap_forwarder rclcpp geometry_msgs px4_msgs nav_msgs)
target_include_directories(mocap_forwarder PRIVATE
  include
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS
  mocap_forwarder
  uam_handler
  DESTINATION lib/${PROJECT_NAME})

ament_package()
