cmake_minimum_required(VERSION 3.17)
project(gather_elements CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings")

set(CUDA_PATH "/usr/local/cuda")
set(TORCH_LIB_PATH "/home/ang/miniconda3/envs/ml/lib/python3.6/site-packages/torch/lib")
set(TRT_PATH "/home/ang/TensorRT-7.0.0.11")

include_directories(
        .
        /home/ang/miniconda3/envs/ml/lib/python3.6/site-packages/torch/include/torch/csrc/api/include
        /home/ang/miniconda3/envs/ml/lib/python3.6/site-packages/torch/include
        ${CUDA_PATH}/include
        ${TRT_PATH}/include
)

add_executable(gather_elements main.cpp gather_elements.cuh gather_elements.cu)

target_link_libraries(
        gather_elements PRIVATE
        ${TORCH_LIB_PATH}/libtorch_cpu.so
        ${TORCH_LIB_PATH}/libtorch_cuda.so
        ${TORCH_LIB_PATH}/libc10_cuda.so
        ${TORCH_LIB_PATH}/libc10.so
        ${TRT_PATH}/lib/libnvinfer.so
        ${TRT_PATH}/lib/libnvinfer_plugin.so
        ${TRT_PATH}/lib/libnvparsers.so
)

set_target_properties(
        gather_elements
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)