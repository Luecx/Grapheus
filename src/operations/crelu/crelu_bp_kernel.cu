
#include "crelu_bp.h"

// clang-format off
__global__ void operations::crelu_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    float max,
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

    if (B[idb] > 0 && B[idb] < max) {
        A_grd[ida] = B_grd[idb];
    } else {
        A_grd[ida] = 0;
    }
}

// a fast version which assumes leading dimension of both matrices is m
// to be precise this assumes that the matrix itself is not a submatrix
// clang-format off
__global__ void operations::crelu_bp_kernel_fast(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    float max,
    size_t size){
    // clang-format on

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > size) return;

    if (B[idx] > 0 && B[idx] < max) {
        A_grd[idx] = B_grd[idx];
    } else {
        A_grd[idx] = 0;
    }
}