cmake_minimum_required(VERSION 3.24)
project(rowwise_softmax LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 90)
add_executable(rowwise_softmax_test
  src/rowwise_softmax.cu
  tests/rowwise_softmax_test.cu)
target_include_directories(rowwise_softmax_test PRIVATE include)
target_compile_features(rowwise_softmax_test PRIVATE cxx_std_17)
