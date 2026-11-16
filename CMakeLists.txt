cmake_minimum_required(VERSION 3.20)
project(forge_native CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch for C++ extensions
find_package(Torch REQUIRED)

# Native FORGE kernel library
add_library(forge_kernels SHARED
    native/src/fused_relu_linear_backward.cpp
)
target_include_directories(forge_kernels PUBLIC native/include)
target_link_libraries(forge_kernels ${TORCH_LIBRARIES})
set_property(TARGET forge_kernels PROPERTY CXX_STANDARD 20)
