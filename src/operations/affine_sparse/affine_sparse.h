#ifndef GRAPHEUS_AFFINE_SPARSE_H
#define GRAPHEUS_AFFINE_SPARSE_H

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"

namespace operations{

// clang-format off
__global__ void affine_sparse_kernel(
    const float*        __restrict__ mat,
    const size_t*       __restrict__ inp_col_indices,
    const size_t                     inp_col_max_entries,
    const float*        __restrict__ bia,
          float*        __restrict__ res,
    const size_t                     m,
    const size_t                     n,
    const size_t                     lda,
    const size_t                     ldc);
// clang-format on

/**
 * Performs sparse matrix multiplication followed by bias addition:
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
inline void affine_sparse(
    data::DenseMatrix<float>& wgt,
    data::SparseMatrix      & inp,
    data::DenseMatrix<float>& bia,
    data::DenseMatrix<float>& res){

    auto M = wgt.m;
    auto B = inp.n;

    ASSERT(bia.m == M)
    ASSERT(bia.n == 1)
    ASSERT(res.m == M)
    ASSERT(res.n == B)

    ASSERT(wgt.first<DEV>())
    ASSERT(inp.values.address<DEV>())
    ASSERT(bia.first<DEV>())
    ASSERT(res.first<DEV>())


    if(data::is_gpu(DEV)){
        // TODO tune
        constexpr int block_size_x = 1;
        constexpr int block_size_y = 128;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)res.n / block_size_x),
                   std::ceil((float)res.m / block_size_y));

        affine_sparse_kernel<<<grid, block>>>(
            wgt.first<DEV>(),
            inp.values.address<DEV>(),
            inp.max_entries_per_column,
            bia.first<DEV>(),
            res.first<DEV>(),
            M,B,
            wgt.ld,
            res.ld);

    }else{
        ASSERT(false)
    }
}

}

#endif    // GRAPHEUS_AFFINE_SPARSE_H
