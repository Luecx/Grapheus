#include "affine_sparse.h"
// clang-format off
__global__ void operations::affine_sparse_kernel(
    const float*        __restrict__ mat,
    const size_t*       __restrict__ inp_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ bia,
          float*        __restrict__ res,
    const size_t                     m,
    const size_t                     n,
    const size_t                     lda,
    const size_t                     ldc){

    // clang-format on
    // compute which output value we are looking at
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ size_t s_col_id[128];
    // copy column indices
    if (threadIdx.y <= inp_col_max_entries){
        // get the offset at which we look into our sparse input
        int offset = col * (inp_col_max_entries + 1);
        // check how many values we are going to read
        s_col_id[threadIdx.y] = inp_col_indices[offset + threadIdx.y];
    }

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    __syncthreads();

    // track the sum
    float sum = bia[row];

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = 1; i < 1 + s_col_id[0]; i++) {

        // get the sparse index (set row of the input)
        auto b_row = s_col_id[i];
        // get the corresponding weight
        auto wgt = mat[MATRIX_INDEX(lda, row, b_row)];

        sum += wgt;
    }
    res[MATRIX_INDEX(ldc, row, col)] = sum;
};