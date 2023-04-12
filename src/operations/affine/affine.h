//
// Created by finne on 08.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"

namespace operations {

/**
 * @brief Adds a bias vector to each row of a matrix.
 *
 * This function adds the elements of a bias vector to the corresponding elements
 * of each row of a matrix. Specifically, for each row i and column j of the matrix,
 * this function computes:
 *
 *     res[i,j] += vec[i]
 *
 * where `vec` is the bias vector and `res` is the matrix to which the bias is added.
 *
 * @param vec Pointer to the bias vector, which must have size at least `m`.
 * @param res Pointer to the matrix to which the bias is added, which must have dimensions `m` x `n`.
 * @param m The number of rows in the matrix.
 * @param n The number of columns in the matrix.
 * @param ld_res The leading dimension of the matrix (i.e., the number of rows between consecutive
 * columns).
 */
__global__ void add_bias_kernel(const float* __restrict__ vec,
                                float* __restrict__ res,
                                int m,
                                int n,
                                int ld_res) {
    // clang-format on
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
//
//    __shared__ float bia_data[16];
//    if(threadIdx.y < 16){
//        bia_data[threadIdx.y] = vec[idy];
//    }
//    __syncthreads();
//
//    if (idx >= n || idy >= m)
//        return;
//    res[MATRIX_INDEX(ld_res, idy, idx)] += bia_data[threadIdx.y];

    if (idx >= n || idy >= m)
        return;
    res[MATRIX_INDEX(ld_res, idy, idx)] += vec[idy];
}

/**
 * Applies an affine transformation to the input matrix, using the weight matrix and bias vector
 * provided, and stores the result in the output matrix. The operation is performed on the GPU if the
 * input matrices are on the GPU.
 *
 * @tparam DEV the device type (CPU or GPU) on which the operation is performed
 * @param[in] inp the input matrix to be transformed
 * @param[in] wgt the weight matrix used for the transformation
 * @param[in] bia the bias vector added to the result of the transformation
 * @param[out] out the output matrix where the transformed result is stored
 *
 * @note that only GPU is currently supported for this operation
 */
// clang-format off
template<data::Device DEV>
inline void affine(const data::DenseMatrix<float>& inp,
                   const data::DenseMatrix<float>& wgt,
                   const data::DenseMatrix<float>& bia,
                         data::DenseMatrix<float>& out) {
    // clang-format on
    if (!data::is_gpu(DEV))
        ERROR(false);

    // 1. perform wgt * inp = out
    const float alpha = 1;
    const float beta  = 0;
    // clang-format off
    cublasSgemm(CUBLAS_HANDLE,
                CUBLAS_OP_N, CUBLAS_OP_N,
                out.m, out.n, inp.m,
                &alpha,
                wgt.first<data::GPU>(), wgt.ld,
                inp.first<data::GPU>(), inp.ld, &beta,
                out.first<data::GPU>(), out.ld);
    // clang-format on

    // 2. perform bias addition
    constexpr int block_size_x = 2;
    constexpr int block_size_y = 32;
    // clang-format off
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)out.n / block_size_x),
               std::ceil((float)out.m / block_size_y));

    add_bias_kernel<<<grid, block>>>(bia.first<data::GPU>(),
                                     out.first<data::GPU>(),
                                     out.m,
                                     out.n,
                                     out.ld);
    // clang-format on
}

}    // namespace operations
