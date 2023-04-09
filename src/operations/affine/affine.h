//
// Created by finne on 08.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"

namespace operations {

__global__ void add_bias_kernel(
    const float* __restrict__ vec,
          float* __restrict__ res,
    int m,
    int n,
    int ld_res){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;

    res[MATRIX_INDEX(ld_res, idy, idx)] += vec[idy];
}

// TODO fix leading dimension of bias
// clang-format off
template<data::Device DEV>
inline void affine(const data::DenseMatrix<float>& inp,
                   const data::DenseMatrix<float>& wgt,
                   const data::DenseMatrix<float>& bia,
                         data::DenseMatrix<float>& out) {
    // clang-format on
    if(!data::is_gpu(DEV))
        ERROR(false);

    // 1. perform wgt * inp = out
    const float alpha = 1;
    const float beta  = 0;
    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_N, CUBLAS_OP_N,
                out.m, out.n, inp.m,
                &alpha,
                wgt.first<data::GPU>(), wgt.ld,
                inp.first<data::GPU>(), inp.ld, &beta,
                out.first<data::GPU>(), out.ld);

    // 2. perform bias addition
    constexpr int block_size_x = 2;
    constexpr int block_size_y = 32;
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)out.n / block_size_x),
               std::ceil((float)out.m / block_size_y));

    add_bias_kernel<<<grid, block>>>(bia.first<data::GPU>(),
                                     out.first<data::GPU>(),
                                     out.m,
                                     out.n,
                                     out.ld);
}

}    // namespace operations
