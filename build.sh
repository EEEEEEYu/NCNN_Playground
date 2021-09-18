export VULKAN_SDK=`pwd`/vulkansdk-macos-1.2.162.0/macOS

cmake --build /Applications/代码/C++/NCNN_Playground/cmake-build-release --target NCNN_Playground -- -j 9 \
    -DCMAKE_OSX_ARCHITECTURES="x86_64" \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/libomp.a" \
    -DVulkan_INCLUDE_DIR=`pwd`/vulkansdk-macos-1.2.162.0/MoltenVK/include \
    -DVulkan_LIBRARY=`pwd`/vulkansdk-macos-1.2.162.0/MoltenVK/dylib/macOS/libMoltenVK.dylib \
    -DNCNN_VULKAN=ON .