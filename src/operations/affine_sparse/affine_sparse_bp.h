#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"
#include "../affine/affine_bp.h"

namespace operations {

// clang-format off
__global__ void affine_sparse_bp_kernel(      float*  __restrict__ mat_grd,
                                        const size_t* __restrict__ inp_col_indices,
                                        const size_t               inp_col_max_entries,
                                              float*  __restrict__ bia_grd,
                                        const float*  __restrict__ res,
                                        const float*  __restrict__ res_grd,
                                        const size_t m,
                                        const size_t n,
                                        const size_t lda,
                                        const size_t ldc,
                                        const float  ft_regularization);
// clang-format on

/**
 * Performs backpropagation for following sparse matrix multiplication followed by bias addition:
 * res = matrix * inp + bia
 *
 * The input matrix is sparse and the output matrix is dense.
 *
 * @param matrix    The dense matrix with dimensions [M, N]
 * @param inp       The sparse matrix with dimensions [N, B]
 * @param bia       The bias with dimensions [M, 1]
 * @param res       The output matrix with dimensions [M, B]
 *
 * @tparam DEV     The device on which to perform the operation
 */
template<data::Device DEV>
inline void affine_sparse_bp(data::DenseMatrix<float>& wgt_grd,
                             data::SparseMatrix&       inp,
                             data::DenseMatrix<float>& bia_grd,
                             data::DenseMatrix<float>& res,
                             data::DenseMatrix<float>& res_grd,
                             const float               ft_regularization) {

    auto M = wgt_grd.m;
    auto B = inp.n;

    ASSERT(bia_grd.m == M)
    ASSERT(bia_grd.n == 1)
    ASSERT(res_grd.m == M)
    ASSERT(res_grd.n == B)

    ASSERT(wgt_grd.first<DEV>())
    ASSERT(inp.values.address<DEV>())
    ASSERT(bia_grd.first<DEV>())
    ASSERT(res_grd.first<DEV>())

    if (data::is_gpu(DEV)) {
        // TODO tune
        constexpr int block_size_x = 1;
        constexpr int block_size_y = 128;

        dim3          block(block_size_x, block_size_y);
        dim3          grid(std::ceil((float) res_grd.n / block_size_x),
                  std::ceil((float) res_grd.m / block_size_y));

        affine_sparse_bp_kernel<<<grid, block>>>(wgt_grd.first<DEV>(),
                                                 inp.values.address<DEV>(),
                                                 inp.max_entries_per_column,
                                                 bia_grd.first<DEV>(),
                                                 res.first<DEV>(),
                                                 res_grd.first<DEV>(),
                                                 M,
                                                 B,
                                                 wgt_grd.ld,
                                                 res_grd.ld,
                                                 ft_regularization);

        // Unused as Bias being updated in above bp
        // dim3 block2(128, 8);
        // dim3 grid2(std::ceil((float) res_grd.n / 128), std::ceil((float) res_grd.m / 8));

        // reduce_row<<<grid2, block2>>>(res_grd.first<DEV>(),
        //                               bia_grd.first<DEV>(),
        //                               res_grd.m,
        //                               res_grd.n,
        //                               res_grd.ld);
    } else {
        ASSERT(false)
    }
}

}    // namespace operations
