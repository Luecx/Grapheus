cmake_minimum_required(VERSION 3.17)
project(Grapheus VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDA)
find_package(Threads)
find_package(OpenMP)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)

add_executable(Grapheus ${SRCS} src/main.cu)

set_property(TARGET Grapheus PROPERTY CUDA_ARCHITECTURES OFF)
set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

set_target_properties(Grapheus PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Suppress specific NVCC warnings
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-march=native \
    -Xcudafe --diag_suppress=68 \
    -Xcudafe --diag_suppress=20050")

target_link_libraries(Grapheus ${CUDA_LIBRARIES})
target_link_libraries(Grapheus ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(Grapheus ${CUDA_cusparse_LIBRARY})
target_link_libraries(Grapheus ${CMAKE_THREAD_LIBS_INIT})