cmake_minimum_required(VERSION 3.17)
project(NCNN_Playground)

set(CMAKE_CXX_STANDARD 11)
add_executable(NCNN_Playground main.cpp)

set(Vulkan_INCLUDE_DIR "/Applications/代码/C++/ncnn/vulkansdk-macos-1.2.162.0/MoltenVK/include")
set(Vulkan_LIBRARY "/Applications/代码/C++/ncnn/vulkansdk-macos-1.2.162.0/MoltenVK/dylib/macOS/libMoltenVK.dylib")

set(ncnn_DIR "/Applications/代码/C++/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(NCNN_Playground ncnn)