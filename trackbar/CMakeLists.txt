cmake_minimum_required(VERSION 3.1)
project(submission_trackbar)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

add_executable(${PROJECT_NAME} submission.cpp)

target_link_libraries (
        ${PROJECT_NAME}
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
)
