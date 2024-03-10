#include "affine_sparse_bp.h"

// clang-format off
__global__ void operations::affine_sparse_bp_kernel(      float*  __restrict__ mat_grd,
                                                    const size_t* __restrict__ inp_col_indices,
                                                    const size_t               inp_col_max_entries,
                                                          float*  __restrict__ bia_grd,
                                                    const float*  __restrict__ res,
                                                    const float*  __restrict__ res_grd,
                                                    const size_t m,
                                                    const size_t n,
                                                    const size_t lda,
                                                    const size_t ldc,
                                                    const float  ft_regularization) {
    // clang-format on

    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    // get the offset at which we look into our sparse input
    const int offset = col * (inp_col_max_entries + 1);
    // check how many values we are going to read
    const int count     = inp_col_indices[offset];

    float     res_grd_v = res_grd[MATRIX_INDEX(ldc, row, col)];

    res_grd_v += ft_regularization * (res[MATRIX_INDEX(ldc, row, col)] > 0.0);

    if (res_grd_v == 0)
        return;

    atomicAdd(&bia_grd[row], res_grd_v);

    // start at offset + 1 (offset contains the amount of values to read)
    for (int i = offset + 1; i < offset + 1 + count; i++) {
        // get the sparse index (set row of the input)
        auto b_row = inp_col_indices[i];
        // get the corresponding weight
        atomicAdd(&mat_grd[MATRIX_INDEX(lda, row, b_row)], res_grd_v);
    }
};