#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"

namespace operations{

__global__ void affine_sparse_shared_kernel(
        const float*        __restrict__ mat,
        const size_t      * __restrict__ inp1_col_indices,
        const size_t      * __restrict__ inp2_col_indices,
        const size_t                     inp_col_max_entries,
        const float*        __restrict__ bia,
              float*        __restrict__ res,
        const size_t                     m,
        const size_t                     n,
        const size_t                     ldw,
        const size_t                     ldo);

template<data::Device DEV>
inline void affine_sparse_shared(
    data::DenseMatrix<float>& wgt,
    data::SparseMatrix      & inp1,
    data::SparseMatrix      & inp2,
    data::DenseMatrix<float>& bia,
    data::DenseMatrix<float>& res){

    auto M = wgt.m;
    auto B = inp1.n;

    ASSERT(inp1.n == B);
    ASSERT(inp2.n == B);
    ASSERT(res.n  == B);

    ASSERT(bia.n  == 1)

    ASSERT(res.m == M * 2);
    ASSERT(bia.m == M * 2);

    ASSERT(wgt.first<DEV>())
    ASSERT(inp1.values.address<DEV>())
    ASSERT(inp2.values.address<DEV>())
    ASSERT(bia.first<DEV>())
    ASSERT(res.first<DEV>())

    if(data::is_gpu(DEV)){
        // TODO tune
        constexpr int block_size_x = 1;
        constexpr int block_size_y = 128;

        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)res.n / block_size_x),
                   std::ceil((float)res.m / block_size_y));

        affine_sparse_shared_kernel<<<grid, block>>>(
            wgt.first<DEV>(),
            inp1.values.address<DEV>(),
            inp2.values.address<DEV>(),
            inp1.max_entries_per_column,
            bia.first<DEV>(),
            res.first<DEV>(),
            M,B,
            wgt.ld,
            res.ld);
    }else{
        ERROR(false)
    }
}

}