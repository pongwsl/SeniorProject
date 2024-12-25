// File: test6.cpp
// Description: Benchmarks MediaPipe's hand tracking on CPU using images from the data folder.

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// MediaPipe and OpenCV headers
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"

// OpenCV for image handling
#include <opencv2/opencv.hpp>

// For directory traversal
#include <filesystem>

namespace fs = std::filesystem;

// Function to load MediaPipe graph config
mediapipe::CalculatorGraphConfig LoadGraphConfig(const std::string& config_path) {
    mediapipe::CalculatorGraphConfig config;
    MP_CHECK_OK(mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(config_path, &config));
    return config;
}

// Function to print normalized landmarks
void PrintNormalizedLandmarks(const mediapipe::NormalizedLandmarkList& landmarks) {
    std::cout << "Normalized Hand Landmarks:\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Landmark\tX\t\tY\t\tZ\n";
    std::cout << "-------------------------------------------\n";
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const auto& lm = landmarks.landmark(i);
        std::cout << i << "\t\t" << lm.x() << "\t\t" << lm.y() << "\t\t" << lm.z() << "\n";
    }
    std::cout << "-------------------------------------------\n";
}

// Function to print world landmarks
void PrintWorldLandmarks(const mediapipe::LandmarkList& landmarks) {
    std::cout << "World Hand Landmarks (in millimeters):\n";
    std::cout << "-------------------------------------------\n";
    std::cout << "Landmark\tX(mm)\t\tY(mm)\t\tZ(mm)\n";
    std::cout << "-------------------------------------------\n";
    for (int i = 0; i < landmarks.landmark_size(); ++i) {
        const auto& lm = landmarks.landmark(i);
        std::cout << i << "\t\t" << lm.x() * 1000 << "\t\t" << lm.y() * 1000 << "\t\t" << lm.z() * 1000 << "\n";
    }
    std::cout << "-------------------------------------------\n";
}

int main(int argc, char** argv) {
    // Path to the MediaPipe graph config file
    const std::string kGraphConfigPath = "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";

    // Initialize the MediaPipe graph
    mediapipe::CalculatorGraphConfig config = LoadGraphConfig(kGraphConfigPath);
    mediapipe::CalculatorGraph graph;
    mediapipe::Status run_status = graph.Initialize(config);
    if (!run_status.ok()) {
        std::cerr << "Failed to initialize the graph: " << run_status.message();
        return EXIT_FAILURE;
    }

    // Start running the graph
    run_status = graph.StartRun({});
    if (!run_status.ok()) {
        std::cerr << "Failed to start the graph: " << run_status.message();
        return EXIT_FAILURE;
    }

    // Path to the data folder containing images
    const std::string data_folder = "data";

    // Supported image extensions
    std::vector<std::string> extensions = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"};

    // Collect image paths
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(data_folder)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    if (image_paths.empty()) {
        std::cerr << "No PNG or JPG/JPEG images found in the data folder: " << data_folder << std::endl;
        return EXIT_FAILURE;
    }

    // Variables to store processing times
    std::vector<double> processing_times;

    // Iterate through each image
    for (const auto& img_path : image_paths) {
        std::cout << "\nProcessing Image: " << fs::path(img_path).filename().string() << std::endl;

        // Load the image using OpenCV
        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << img_path << std::endl;
            continue;
        }

        // Convert BGR to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Convert OpenCV image to MediaPipe ImageFrame
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
            mediapipe::ImageFormat::SRGB, image.cols, image.rows,
            mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(*input_frame);
        image.copyTo(input_frame_mat);

        // Serialize the ImageFrame into a packet
        mediapipe::Packet image_packet = mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(0));

        // Send the image packet into the graph
        graph.AddPacketToInputStream("input_video", image_packet);

        // Start time measurement
        auto start_time = std::chrono::high_resolution_clock::now();

        // Get the output packets
        mediapipe::Packet packet;
        bool success = graph.WaitForOutputPacket("hand_landmarks", &packet);
        if (!success) {
            std::cerr << "Failed to get hand landmarks for image: " << img_path << std::endl;
            continue;
        }

        // End time measurement
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> processing_time = end_time - start_time;
        processing_times.push_back(processing_time.count());

        std::cout << "CPU Processing Time: " << processing_time.count() << " seconds" << std::endl;

        // Retrieve hand landmarks
        const mediapipe::NormalizedLandmarkList& normalized_landmarks =
            packet.Get<mediapipe::NormalizedLandmarkList>();

        // Retrieve world landmarks
        // Note: MediaPipe's C++ API may require additional configuration to retrieve world landmarks.
        // This example assumes that world landmarks are available in the packet named "world_landmarks".
        mediapipe::Packet world_packet;
        bool world_success = graph.WaitForOutputPacket("world_landmarks", &world_packet);
        mediapipe::LandmarkList world_landmarks;
        if (world_success) {
            world_landmarks = world_packet.Get<mediapipe::LandmarkList>();
        }

        // Print landmarks
        PrintNormalizedLandmarks(normalized_landmarks);
        if (world_success) {
            PrintWorldLandmarks(world_landmarks);
        }

        // Optional: Draw landmarks on the image (commented out as per requirement)
        /*
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        // Drawing code here using OpenCV
        cv::imshow("Hand Landmarks", image);
        cv::waitKey(0);
        */

        // Release resources for the current image
        graph.CloseInputStream("input_video");
    }

    // Calculate and display summary statistics
    if (!processing_times.empty()) {
        double sum = 0.0;
        double max_time = processing_times[0];
        double min_time = processing_times[0];
        for (const auto& t : processing_times) {
            sum += t;
            if (t > max_time) max_time = t;
            if (t < min_time) min_time = t;
        }
        double avg_time = sum / processing_times.size();

        std::cout << "\n--- CPU Processing Time Summary ---" << std::endl;
        std::cout << "Total Images Processed: " << processing_times.size() << std::endl;
        std::cout << "Average Processing Time: " << avg_time << " seconds" << std::endl;
        std::cout << "Maximum Processing Time: " << max_time << " seconds" << std::endl;
        std::cout << "Minimum Processing Time: " << min_time << " seconds" << std::endl;
    } else {
        std::cout << "\nNo images were processed." << std::endl;
    }

    // Wait until the graph is done processing
    mediapipe::Status run_status = graph.WaitUntilDone();
    if (!run_status.ok()) {
        std::cerr << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}