cmake_minimum_required(VERSION 3.10)

project(colorSpaces)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs videoio)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries (
        ${PROJECT_NAME}
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
)
