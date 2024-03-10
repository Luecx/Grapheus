//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include "start.h"

#include <iostream>

#ifdef NDEBUG
#define ASSERT(expr)
#else
#define ASSERT(expr)                                                                                 \
    {                                                                                                \
        if (!static_cast<bool>(expr)) {                                                              \
            std::cout << "[ASSERT] in expression " << (#expr) << std::endl;                          \
            std::cout << "    file: " << __FILE__ << std::endl;                                      \
            std::cout << "    line: " << __LINE__ << std::endl;                                      \
            std::cout << "    func: " << __FUNCTION__ << std::endl;                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    }
#endif

#define ERROR(expr)                                                                                  \
    {                                                                                                \
        if (!static_cast<bool>(expr)) {                                                              \
            std::cout << "[ERROR] in expression " << (#expr) << std::endl;                           \
            std::cout << "    file: " << __FILE__ << std::endl;                                      \
            std::cout << "    line: " << __LINE__ << std::endl;                                      \
            std::cout << "    func: " << __FUNCTION__ << std::endl;                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    }

#define CUDA_ASSERT(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline void gpuAssert(cublasStatus_t code, const char* file, int line, bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        switch (code) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_NOT_INITIALIZED file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_ALLOC_FAILED:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_ALLOC_FAILED file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_INVALID_VALUE:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_INVALID_VALUE file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_ARCH_MISMATCH:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_ARCH_MISMATCH file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_MAPPING_ERROR:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_MAPPING_ERROR file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_EXECUTION_FAILED:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_EXECUTION_FAILED file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_INTERNAL_ERROR:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_INTERNAL_ERROR file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_NOT_SUPPORTED:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_NOT_SUPPORTED file: %s line: %d ",
                        file,
                        line);
                break;

            case CUBLAS_STATUS_LICENSE_ERROR:
                fprintf(stderr,
                        "cuBLAS Error: CUBLAS_STATUS_LICENSE_ERROR file: %s line: %d ",
                        file,
                        line);
                break;
        }
        if (abort)
            exit(code);
    }
}
