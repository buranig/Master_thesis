cmake_minimum_required(VERSION 3.5)
project(bumper_cars)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

add_subdirectory(lib/model-driving)


# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(bumper_car_model src/driving_dynamics_test.cpp)
target_link_libraries(bumper_car_model PRIVATE driving_model)
# ament_target_dependencies(bumper_car_model rclcpp)

# Install Cpp executables
install(TARGETS
  bumper_car_model
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  bumper_cars/collision_avoidance_ros2.py
  bumper_cars/state_buffer.py
  bumper_cars/classes/Controller.py
  bumper_cars/classes/CarModel.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
)



ament_package()