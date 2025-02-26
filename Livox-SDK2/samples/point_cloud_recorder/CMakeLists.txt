cmake_minimum_required(VERSION 3.0)

set(DEMO_NAME point_cloud_recorder)
add_executable(${DEMO_NAME} main.cpp)

find_package(Boost REQUIRED COMPONENTS filesystem system)
target_link_libraries(${DEMO_NAME}
        PUBLIC
        livox_lidar_sdk_static
)

