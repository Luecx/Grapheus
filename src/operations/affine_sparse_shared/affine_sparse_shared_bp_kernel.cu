
#include "affine_sparse_shared_bp.h"

// clang-format off
__global__ void operations::affine_sparse_shared_bp_kernel(
          float*        __restrict__ mat_grd,
    const size_t      * __restrict__ inp1_col_indices,
    const size_t      * __restrict__ inp2_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ res_grd,
    const size_t                     m,
    const size_t                     n,
    const size_t                     ldw,
    const size_t                     ldo){

    // clang-format on
    // compute which output value we are looking at
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // skip out of bounds
    if (col >= n || row >= m)
        return;

    // get the offset at which we look into our sparse input
    int offset = col * (inp_col_max_entries + 1);

    const size_t* indices[2]{inp1_col_indices, inp2_col_indices};

    for(int j = 0; j < 2; j++){
        // check how many values we are going to read
        const int count = indices[j][offset];

        float res_grd_v = res_grd[MATRIX_INDEX(ldo, row + j * m, col)];
        if(res_grd_v == 0)
            return;

        // start at offset + 1 (offset contains the amount of values to read)
        for (int i = offset + 1; i < offset + 1 + count; i++) {
            // get the sparse index (set row of the input)
            auto b_row = indices[j][i];
            // get the corresponding weight
            atomicAdd(&mat_grd[MATRIX_INDEX(ldw, row, b_row)], res_grd_v);
        }
    }
}
