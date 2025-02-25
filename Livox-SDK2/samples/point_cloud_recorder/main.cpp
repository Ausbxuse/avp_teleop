#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <cstring>
#include <cmath> 
#include <csignal>
#include <iomanip>
#include <sstream>
#include <mutex>

#include "livox_lidar_api.h"
#include "livox_lidar_def.h"

struct PointXYZ {
    float x;
    float y;
    float z;
};

// Global container for point cloud data.
std::vector<PointXYZ> cloud;
std::mutex cloud_mutex;

volatile uint32_t g_lidar_handle = 0;
volatile std::sig_atomic_t g_stop = 0;

// Signal handler for SIGINT (Ctrl+C)
void SignalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\nSIGINT received. Stopping recording..." << std::endl;
        g_stop = 1;
    }
}

std::string GetTimestampFilename() {
    // Get current time as system_clock time_point.
    auto now = std::chrono::system_clock::now();
    // Convert to time_t for formatting.
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    // Get milliseconds (remainder of duration).
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;

    char timeStr[64];
    std::strftime(timeStr, sizeof(timeStr), "%m%d_%H%M%S", std::localtime(&t));

    // Build filename with milliseconds appended.
    std::stringstream ss;
    ss << timeStr << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".pcd";
    return ss.str();
}

// New helper: Save a given vector of points to an ASCII PCD file.
void SavePointsToPCD(const std::vector<PointXYZ>& points, const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
    ofs << "VERSION 0.7\n";
    ofs << "FIELDS x y z\n";
    ofs << "SIZE 4 4 4\n";
    ofs << "TYPE F F F\n";
    ofs << "COUNT 1 1 1\n";
    ofs << "WIDTH " << points.size() << "\n";
    ofs << "HEIGHT 1\n";
    ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
    ofs << "POINTS " << points.size() << "\n";
    ofs << "DATA ascii\n";

    for (const auto& point : points) {
        ofs << point.x << " " << point.y << " " << point.z << "\n";
    }
    ofs.close();
    std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
}

void PointCloudCallback(const uint32_t handle, const uint8_t dev_type, 
                        LivoxLidarEthernetPacket* packet, void* client_data) {
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
    
    // Process the point cloud data based on the point data type.
    // Lock the mutex while appending points.
    std::lock_guard<std::mutex> lock(cloud_mutex);
    if (packet->data_type == kLivoxLidarCartesianCoordinateHighData) {
        // High precision Cartesian: points in mm.
        LivoxLidarCartesianHighRawPoint* points = 
            reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(packet->data);
        for (uint16_t i = 0; i < num_points; ++i) {
            PointXYZ point;
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
    // Note: The saving step is now handled in a separate thread.
}

int main(int argc, const char* argv[]) {
    // Install the SIGINT handler.
    std::signal(SIGINT, SignalHandler);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_path>" << std::endl;
        return -1;
    }

    std::string config_path = argv[1];

    if (!LivoxLidarSdkInit(config_path.c_str())) {
        std::cerr << "Failed to initialize Livox LiDAR SDK!" << std::endl;
        return -1;
    }

    SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox LiDAR SDK!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    std::cout << "Waiting for LiDAR handle discovery..." << std::endl;
    for (int i = 0; i < 50 && g_lidar_handle == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (g_lidar_handle == 0) {
        std::cerr << "Failed to get LiDAR handle from callback." << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    if (EnableLivoxLidarPointSend(g_lidar_handle, nullptr, nullptr) != kLivoxLidarStatusSuccess) {
        std::cerr << "Failed to enable point cloud data sending!" << std::endl;
        LivoxLidarSdkUninit();
        return -1;
    }

    // periodically save .pcd (per every 33ms)
    std::thread saving_thread([](){
        const std::chrono::milliseconds interval(33);
        while (!g_stop) {
            std::this_thread::sleep_for(interval);
            std::vector<PointXYZ> points_to_save;
            {
                std::lock_guard<std::mutex> lock(cloud_mutex);
                // Swap out the current cloud and clear it.
                if (!cloud.empty()) {
                    points_to_save.swap(cloud);
                }
            }
            if (!points_to_save.empty()) {
                std::string filename = GetTimestampFilename();
                SavePointsToPCD(points_to_save, filename);
            }
        }
    // NOTE: for the final period, don't save, since data might not be worth exact 33ms
    });

    std::cout << "Collecting point cloud data. Press Ctrl+C to stop recording." << std::endl;

    // main thread waits until SIGINT is received.
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Clean up: join the saving thread and uninitialize the SDK.
    if (saving_thread.joinable()) {
        saving_thread.join();
    }

    LivoxLidarSdkUninit();
    std::cout << "Livox LiDAR SDK Uninitialized. Recording complete." << std::endl;

    return 0;
}
