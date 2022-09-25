set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

cmake_minimum_required(VERSION 3.21)
project(dtmfCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin")

add_executable(dtmfCUDA dtmfCuda.cu) # main.cu # wavFile.cu

set_target_properties(dtmfCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#set_target_properties(cudaTest PROPERTIES CUDA_ARCHITECTURES "53")

#target_compile_options(cudaTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#        --default-stream per-thread
#>)