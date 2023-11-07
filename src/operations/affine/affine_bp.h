//
// Created by finne on 08.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"
#include "../gradient_operation.h"

namespace operations {
__device__ void inline reduce_warp(volatile float* sdata, size_t tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ inline void reduce_row(const float* mat,
                                        float* res,
                                  const int m,
                                  const int n,
                                  const int ld) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x * 2;

    if(row >= m || col >= n) return;

    // avoid parallel reduction remainder is not divisible by block dimension
    if((blockIdx.x + 1) * blockDim.x * 2 > n){
        int id1 = col;
        int id2 = col + blockDim.x;

        if(id1 < n)
            atomicAdd(&res[row], mat[MATRIX_INDEX(ld, row, id1)]);
        if(id2 < n)
            atomicAdd(&res[row], mat[MATRIX_INDEX(ld, row, id2)]);
    }else{
        // each block has 512 threads in n direction
        __shared__ float sdata[8][128];

        // load the matrix into shared memory
        sdata[threadIdx.y][threadIdx.x] = mat[MATRIX_INDEX(ld, row, col)]
                                        + mat[MATRIX_INDEX(ld, row, col + blockDim.x)];
        __syncthreads();

        // do reduction
        for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.y][threadIdx.x] += sdata[threadIdx.y][threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x < 32) reduce_warp(sdata[threadIdx.y], threadIdx.x);

        if(threadIdx.x == 0)
            atomicAdd(&res[row], sdata[threadIdx.y][0]);
    }
}

/**
 * @brief Computes the backward pass for an affine layer using GPU acceleration.
 *
 * Given input data, weight matrix, bias vector, and gradients of output data, this function
 * computes gradients of input data, weight matrix, and bias vector for the backward pass of an
 * affine layer in a neural network using GPU acceleration. The computation is performed using
 * CUDA APIs for parallel reduction and matrix multiplication.
 *
 * @tparam DEV The device where the computation is performed (CPU or GPU).
 * @param inp Input data matrix of shape (m, n), stored in row-major order.
 * @param inp_grd Matrix to store gradients of input data of shape (m, n), stored in row-major order.
 * @param wgt Weight matrix of shape (k, n), stored in row-major order.
 * @param wgt_grd Matrix to store gradients of weight matrix of shape (k, n), stored in row-major
 * order.
 * @param bia_grd Vector to store gradients of bias vector of shape (k), stored in row-major order.
 * @param out_grd Gradients of output data matrix of shape (m, k), stored in row-major order.
 *
 * @note Currently only GPU as device is supported
 */
// clang-format off
template<data::Device DEV>
inline void affine_bp(const data::DenseMatrix<float>& inp,
                            data::DenseMatrix<float>& inp_grd,
                      const data::DenseMatrix<float>& wgt,
                            data::DenseMatrix<float>& wgt_grd,
                            data::DenseMatrix<float>& bia_grd,
                            data::DenseMatrix<float>& out_grd,
                      GradientOperation grad_operation = SET) {
    // clang-format on
    if (!data::is_gpu(DEV))
        ERROR(false);

    // beta = 0 indicates that we dont add to the gradients but simply replace them
    const float alpha = 1;
    const float beta  = grad_operation == SET ? 0:1;

    // step 1. Compute gradients of bias using a parallel reduction
    constexpr int block_size_x = 128;
    constexpr int block_size_y = 8;
    dim3          block(block_size_x, block_size_y);
    dim3          grid(std::ceil((float) out_grd.n / block_size_x),
                       std::ceil((float) out_grd.m / block_size_y));

//    cudaMemset(bia_grd.first<DEV>(), 0, sizeof(float) * bia_grd.m);
    reduce_row<<<grid, block>>>(out_grd.first<DEV>(),
                                bia_grd.first<DEV>(),
                                out_grd.m,
                                out_grd.n,
                                out_grd.ld);

    // step 2. compute gradients of weights
    // wgt_grd = inp * out_grd^T
    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                wgt_grd.m,
                wgt_grd.n,
                out_grd.n,
                &alpha,
                out_grd.first<data::GPU>(),
                out_grd.ld,
                inp.first<data::GPU>(),
                inp.ld,
                &beta,
                wgt_grd.first<data::GPU>(),
                wgt_grd.ld);

    // step 3. compute gradients of inputs
    // inp_grd = wgt^T * out_grd
    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                inp_grd.m,
                inp_grd.n,
                wgt.m,
                &alpha,
                wgt.first<data::GPU>(),
                wgt.ld,
                out_grd.first<data::GPU>(),
                out_grd.ld,
                &beta,
                inp_grd.first<data::GPU>(),
                inp_grd.ld);
}

}    // namespace operations
