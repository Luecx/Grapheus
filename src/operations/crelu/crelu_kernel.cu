
#include "crelu.h"

// clang-format off
__global__ void operations::crelu_kernel(
    const float* __restrict__ A,
          float* __restrict__ B,
    float ceil,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    int ida = MATRIX_INDEX(lda, idy, idx);
    int idb = MATRIX_INDEX(ldb, idy, idx);

    B[idb] = max(0.0f, min(ceil, A[ida]));
}

// a fast version which assumes leading dimension of both matrices is m
// to be precise this assumes that the matrix itself is not a submatrix
// clang-format off
__global__ void operations::crelu_kernel_fast(
    const float* __restrict__ A,
          float* __restrict__ B,
    float ceil,
    size_t size){
    // clang-format on

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > size) return;

    B[idx] = max(0.0f, min(ceil, A[idx]));

}