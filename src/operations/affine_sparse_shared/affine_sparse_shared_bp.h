#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"
#include "../affine/affine_bp.h"

namespace operations {

// clang-format off
__global__ void affine_sparse_shared_bp_kernel(
          float*        __restrict__ mat_grd,
    const size_t      * __restrict__ inp1_col_indices,
    const size_t      * __restrict__ inp2_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ res_grd,
    const size_t                     m,
    const size_t                     n,
    const size_t                     ldw,
    const size_t                     ldo);
// clang-format on

// clang-format off
__global__ inline void reduce_row_twice(const float* mat,
                                              float* res,
                                        const int m_res,
                                        const int n_res,
                                        const int ld) {
    // clang-format on
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // always add two values together to begin with
    __shared__ float sdata[8][128];
    if (row >= m_res || col >= n_res) {
        sdata[threadIdx.y][threadIdx.x] = 0;
    } else {
        sdata[threadIdx.y][threadIdx.x] =
            mat[MATRIX_INDEX(ld, row, col)] + mat[MATRIX_INDEX(ld, row + m_res, col)];
    }

    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y][threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
        reduce_warp(sdata[threadIdx.y], threadIdx.x);

    if (threadIdx.x == 0)
        atomicAdd(&res[row], sdata[threadIdx.y][0]);
}


// clang-format off
template<data::Device DEV>
inline void affine_sparse_shared_bp(data::DenseMatrix<float>& wgt_grd,
                                    data::SparseMatrix&       inp1,
                                    data::SparseMatrix&       inp2,
                                    data::DenseMatrix<float>& bia_grd,
                                    data::DenseMatrix<float>& res_grd) {
    // clang-format on

    auto M = wgt_grd.m;
    auto B = inp1.n;

    ASSERT(inp1.n == B);
    ASSERT(inp2.n == B);
    ASSERT(res_grd.n == B);

    ASSERT(bia_grd.n == 1)

    ASSERT(res_grd.m == M * 2);
    ASSERT(bia_grd.m == M * 2);

    ASSERT(inp1.values.address<DEV>())
    ASSERT(inp2.values.address<DEV>())
    ASSERT(bia_grd.first<DEV>())
    ASSERT(res_grd.first<DEV>())

    if (data::is_gpu(DEV)) {
        ERROR(false);
        // DONT USE THIS CODE, its slow

        //        // TODO tune
        //        constexpr int block_size_x = 1;
        //        constexpr int block_size_y = 128;
        //
        //        dim3 block(block_size_x, block_size_y);
        //        dim3 grid (std::ceil((float)res_grd.n / block_size_x),
        //                   std::ceil((float)res_grd.m / block_size_y));
        //
        //        affine_sparse_shared_bp_kernel<<<grid, block>>>(
        //            wgt_grd.first<DEV>(),
        //            inp1.values.address<DEV>(),
        //            inp2.values.address<DEV>(),
        //            inp1.max_entries_per_column,
        //            res_grd.first<DEV>(),
        //            M,B,
        //            wgt_grd.ld,
        //            res_grd.ld);
        //        dim3 block2(128, 8);
        //        dim3 grid2  (std::ceil((float) res_grd.n / 128),
        //                     std::ceil((float) res_grd.m / 8));
        //
        //        cudaMemset(bia_grd.first<DEV>(), 0, sizeof(float) * bia_grd.m);
        //        reduce_row_twice<<<grid2, block2>>>(res_grd.first<DEV>(),
        //                                            bia_grd.first<DEV>(),
        //                                            bia_grd.m,
        //                                            res_grd.n,
        //                                            res_grd.ld);

    } else {
        ERROR(false)
    }
}

}    // namespace operations