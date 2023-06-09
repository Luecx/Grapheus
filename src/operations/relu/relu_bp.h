
#pragma once
#include "../../data/sarray.h"
#include "../../data/device.h"
#include "../../data/matrix_dense.h"
#include <iostream>

namespace operations {

void relu_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb);

__global__ void relu_bp_kernel(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb);

__global__ void relu_bp_kernel_fast(
    const float* __restrict__ A,
          float* __restrict__ A_grd,
    const float* __restrict__ B,
    const float* __restrict__ B_grd,
    size_t size);

template<data::Device DEV>
inline void relu_bp(const data::DenseMatrix<float> &A,
                          data::DenseMatrix<float> &A_grd,
                    const data::DenseMatrix<float> &B,
                    const data::DenseMatrix<float> &B_grd){

    ASSERT(A.size() == B.size());
    ASSERT(A    .first<DEV>());
    ASSERT(A_grd.first<DEV>());
    ASSERT(B    .first<DEV>());
    ASSERT(B_grd.first<DEV>());

    if constexpr(data::is_gpu(DEV)){

        // check if any of the two matrices is a submatrix
        if(A.ld != A.m ||
           B.ld != B.m){
            // optimise block sizes depending on output size
            // TODO properly optimise this
            int block_size_x;
            int block_size_y;
            if(A.m > 128){
                block_size_x = 1;
                block_size_y = 32;
            } else if(A.m > 8){
                block_size_x = 32;
                block_size_y = 8;
            } else{
                block_size_x = 512;
                block_size_y = 1;
            };

            dim3 block(block_size_x, block_size_y);
            dim3 grid (std::ceil((float)A.n / block_size_x),
                       std::ceil((float)A.m / block_size_y));
            relu_bp_kernel<<<grid, block>>>(
                A    .first<DEV>(),
                A_grd.first<DEV>(),
                B    .first<DEV>(),
                B_grd.first<DEV>(),
                A.m,
                A.n,
                A.ld,
                B.ld);
        } else{
            // this is the faster method
            // assumes all data is contigious in memory and uses
            // 1d-indexing
            constexpr int block_size = 512;

            dim3 block(block_size);
            dim3 grid (std::ceil((float)(A.m * A.n) / block_size));
            relu_bp_kernel_fast<<<grid, block>>>(
                A    .first<DEV>(),
                A_grd.first<DEV>(),
                B    .first<DEV>(),
                B_grd.first<DEV>(),
                (A.m * A.n));
        }
    }else{
        relu_bp_host(
            A    .first<DEV>(),
            A_grd.first<DEV>(),
            B    .first<DEV>(),
            B_grd.first<DEV>(),
            A.m,
            A.n,
            A.ld,
            B.ld);
    }
}

}

// clang-format on
