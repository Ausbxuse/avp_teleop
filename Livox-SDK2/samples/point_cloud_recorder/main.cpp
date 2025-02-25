#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <cmath>   // for sinf, cosf
#include <csignal> // for signal handling

#include "livox_lidar_api.h"
#include "livox_lidar_def.h"

// Define a simple point structure for our custom PCD writer.
struct PointXYZ {
    float x;
    float y;
    float z;
};

// Global container for point cloud data.
std::vector<PointXYZ> cloud;

// Global variable to hold the LiDAR handle (set from the callback).
// Declared as volatile because it is modified asynchronously.
volatile uint32_t g_lidar_handle = 0;

// Global flag to indicate that the user wants to stop recording.
volatile std::sig_atomic_t g_stop = 0;

// Signal handler for SIGINT (Ctrl+C)
void SignalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nSIGINT received. Stopping recording..." << std::endl;
        g_stop = 1;
    }
}

// Callback function matching the SDK's expected signature.
// According to the header, the callback is defined as:
//   typedef void (*LivoxLidarPointCloudCallBack)(const uint32_t handle, const uint8_t dev_type,
//                                               LivoxLidarEthernetPacket* data, void* client_data);
void PointCloudCallback(const uint32_t handle, const uint8_t dev_type, 
                        LivoxLidarEthernetPacket* packet, void* client_data) {
    // Capture the first handle we see.
    if (g_lidar_handle == 0) {
        g_lidar_handle = handle;
        std::cout << "Discovered LiDAR handle: " << g_lidar_handle << std::endl;
    }

    std::cout << "Received packet from LiDAR handle: " << handle
              << ", dev_type: " << static_cast<int>(dev_type) << std::endl;
    
    if (!packet || !packet->data) {
        std::cerr << "Packet data is null!" << std::endl;
        return;
    }
    
    // The number of points is provided by the 'dot_num' field.
    uint16_t num_points = packet->dot_num;
    std::cout << "Number of points in packet: " << num_points << std::endl;
    
    // Process the point cloud data based on the point data type.
    if (packet->data_type == kLivoxLidarCartesianCoordinateHighData) {
        // High precision Cartesian: points in mm.
        LivoxLidarCartesianHighRawPoint* points = 
            reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            PointXYZ point;
            // Optionally, convert from mm to meters by dividing by 1000.0f.
            point.x = static_cast<float>(points[i].x);
            point.y = static_cast<float>(points[i].y);
            point.z = static_cast<float>(points[i].z);
            cloud.push_back(point);
        }
    }
    else if (packet->data_type == kLivoxLidarCartesianCoordinateLowData) {
        // Low precision Cartesian: points in cm.
        LivoxLidarCartesianLowRawPoint* points = 
            reinterpret_cast<LivoxLidarCartesianLowRawPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            PointXYZ point;
            point.x = static_cast<float>(points[i].x);
            point.y = static_cast<float>(points[i].y);
            point.z = static_cast<float>(points[i].z);
            cloud.push_back(point);
        }
    }
    else if (packet->data_type == kLivoxLidarSphericalCoordinateData) {
        // Spherical coordinates: conversion to Cartesian is needed.
        LivoxLidarSpherPoint* points = 
            reinterpret_cast<LivoxLidarSpherPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            PointXYZ point;
            float depth = static_cast<float>(points[i].depth);
            // Assuming theta and phi are provided in degrees.
            float theta = static_cast<float>(points[i].theta) * (3.14159265358979323846f / 180.0f);
            float phi   = static_cast<float>(points[i].phi)   * (3.14159265358979323846f / 180.0f);
            point.x = depth * sinf(phi) * cosf(theta);
            point.y = depth * sinf(phi) * sinf(theta);
            point.z = depth * cosf(phi);
            cloud.push_back(point);
        }
    }
    else {
        std::cerr << "Unknown point data type: " << static_cast<int>(packet->data_type) << std::endl;
    }
}

// Custom function to save the collected point cloud data as an ASCII PCD file.
void SavePointCloudToPCD(const std::string& filename) {
    if (cloud.empty()) {
        std::cout << "No point cloud data to save!" << std::endl;
        return;
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the PCD header.
    ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
    ofs << "VERSION 0.7\n";
    ofs << "FIELDS x y z\n";
    ofs << "SIZE 4 4 4\n";
    ofs << "TYPE F F F\n";
    ofs << "COUNT 1 1 1\n";
    ofs << "WIDTH " << cloud.size() << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << cloud.size() << "\n";
    ofs << "DATA ascii\n";

    for (const auto& point : cloud) {
        ofs << point.x << " " << point.y << " " << point.z << "\n";
    }

    ofs.close();
    std::cout << "Saved " << cloud.size() << " points to " << filename << std::endl;
}

int main(int argc, const char* argv[]) {
    // Install the SIGINT handler.
    std::signal(SIGINT, SignalHandler);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_path>" << std::endl;
        return -1;
    }

    std::string config_path = argv[1];

    // Initialize the Livox LiDAR SDK.
    if (!LivoxLidarSdkInit(config_path.c_str())) {
        std::cerr << "Failed to initialize Livox LiDAR SDK!" << std::endl;
        return -1;
    }

    // Register the point cloud callback before starting the SDK.
    SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

    // Start the Livox LiDAR SDK.
    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox LiDAR SDK!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    // Wait until a LiDAR handle is discovered via the callback.
    std::cout << "Waiting for LiDAR handle discovery..." << std::endl;
    for (int i = 0; i < 50 && g_lidar_handle == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (g_lidar_handle == 0) {
        std::cerr << "Failed to get LiDAR handle from callback." << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    // Enable point cloud data sending using the discovered handle.
    if (EnableLivoxLidarPointSend(g_lidar_handle, nullptr, nullptr) != kLivoxLidarStatusSuccess) {
        std::cerr << "Failed to enable point cloud data sending!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    std::cout << "Collecting point cloud data. Press Ctrl+C to stop recording." << std::endl;

    // Run until the user sends a SIGINT (Ctrl+C).
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // "Finally" block: Save data and clean up.
    SavePointCloudToPCD("livox_pointcloud.pcd");
    LivoxLidarSdkUninit();
    std::cout << "Livox LiDAR SDK Uninitialized. Recording complete." << std::endl;

    return 0;
}
