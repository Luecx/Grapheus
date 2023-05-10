//
// Created by Luecx on 18.03.2023.
//
#include "start.h"
#include <iostream>

cublasHandle_t CUBLAS_HANDLE;
cusparseHandle_t CUSPARSE_HANDLE;


void display_header() {
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "C++ version:    v" << __cplusplus << std::endl;
    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        // clang-format off
        std::cout << (i+1)       << ": "
                  << props.name  << ": "
                  << props.major << "."
                  << props.minor << std::endl;
        std::cout << "  Global memory:          " << props.totalGlobalMem    / mb << "mb" << std::endl;
        std::cout << "  Shared memory:          " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory:        " << props.totalConstMem     / kb << "kb" << std::endl;
        std::cout << "  Block registers:        " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:              " << props.warpSize                 << std::endl;
        std::cout << "  Threads per block:      " << props.maxThreadsPerBlock       << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
                                                  << props.maxThreadsDim[1] << ", "
                                                  << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize  [0] << ", "
                                                  << props.maxGridSize  [1] << ", "
                                                  << props.maxGridSize  [2] << " ]" << std::endl;
        // clang-format on
        std::cout << std::endl;
    }

    std::cout << "choosing device: " << 1 << std::endl;
    cudaSetDevice(0);
}

void init() {
    cublasCreate(&CUBLAS_HANDLE);
    // cusparseCreate(&CUSPARSE_HANDLE);
    display_header();
}

void close() {
    cublasDestroy(CUBLAS_HANDLE);
    // cusparseDestroy(CUSPARSE_HANDLE);
}