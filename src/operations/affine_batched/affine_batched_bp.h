//
// Created by finne on 09.04.2023.
//

#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"
#include "../affine/affine_bp.h"
#include "../gradient_operation.h"

namespace operations {


// clang-format off
/**
 * @brief Compute the gradients of the batched affine transformation on the GPU using backpropagation.
 *
 * This function computes the gradients of the weight matrices `wgt` and bias vectors `bia` in the
 * batched affine transformation:
 *
 *     out_i = W_i * x_i + b_i
 *
 * The gradients are computed with respect to the loss in `out_grd`. The function assumes that the
 * input `inp` is the same input that was used to compute the forward pass. The function also
 * assumes that the sub-matrices in the input, weights, biases, and output matrices are evenly spaced.
 *
 * The indexing into each array is done using the leading dimension. The function assumes that all
 * inputs have the same size.
 *
 * The dimensions of the matrices are:
 *
 *     out_grd = [B * m, n]
 *     inp     = [B * k, n]
 *     inp_grd = [B * k, n]
 *     wgt     = [B * m, k]
 *     wgt_grd = [B * m, k]
 *     bia_grd = [B * m, 1]
 *
 * where B is the amount of affine transformations to be performed. The function takes in the
 * matrices inp, wgt, bia, and out_grd and computes the gradients of wgt and bia in wgt_grd and
 * bia_grd, respectively. It also computes the gradient of the input in inp_grd.
 *
 * @tparam DEV the device type, only works with data::GPU
 * @param inp input matrix of shape `inp = [B * k, n]`.
 * @param inp_grd gradient of the input matrix of shape `inp_grd = [B * k, n]`.
 * @param wgt weight matrix of shape `wgt = [B * m, k]`.
 * @param wgt_grd gradient of the weight matrix of shape `wgt_grd = [B * m, k]`.
 * @param bia_grd gradient of the bias vector of shape `bia_grd = [B * m, 1]`.
 * @param out_grd gradient of the output matrix of shape `out_grd = [B * m, n]`.
 * @param batches amount of batches to perform. Also the amount of sub-matrices per matrix
 */
template<data::Device DEV>
inline void affine_batched_bp(const data::DenseMatrix<float>& inp,
                                    data::DenseMatrix<float>& inp_grd,
                              const data::DenseMatrix<float>& wgt,
                                    data::DenseMatrix<float>& wgt_grd,
                                    data::DenseMatrix<float>& bia_grd,
                              const data::DenseMatrix<float>& out_grd,
                              const size_t batches,
                              GradientOperation grad_operation = SET) {

    const size_t m = out_grd.m / batches;
    const size_t n = out_grd.n;
    const size_t k = wgt.n;

    // make sure we can access the pointers at least
    ASSERT(inp    .is_allocated<DEV>());
    ASSERT(wgt    .is_allocated<DEV>());
    ASSERT(wgt_grd.is_allocated<DEV>());
    ASSERT(bia_grd.is_allocated<DEV>());
    ASSERT(out_grd.is_allocated<DEV>());
    ASSERT(inp_grd.is_allocated<DEV>());

    // also make sure we have the same amount of inputs, weights, biases and outputs
    ASSERT(inp.m == inp_grd.m && inp.n == inp_grd.n);
    ASSERT(wgt.m == wgt_grd.m && wgt.n == wgt_grd.n);
    ASSERT(bia_grd.m == out_grd.m &&
           bia_grd.m == wgt_grd.m);
    ASSERT(out_grd.n == inp.n);
    ASSERT(bia_grd.n == 1);
    // clang-format on

    // make sure we have no invalid leading dimensions
    ASSERT(inp.ld && wgt.ld && out_grd.ld);

    // step 1: reduce rows to get the gradient of bias
    constexpr int block_size_x = 128;
    constexpr int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)n               / block_size_x / 2),
               std::ceil((float)m * batches     / block_size_y));
//    cudaMemset(bia_grd.first<DEV>(), 0, sizeof(float) * m * batches);
    reduce_row<<<grid, block>>>(
        out_grd.first<DEV>(),
        bia_grd.first<DEV>(),
        m * batches,
        n,
        out_grd.ld);

    const float alpha = 1;
    const float beta  = grad_operation == SET ? 0:1;
    // step 2. compute gradients of weights
    // wgt_grd = out_grd * inp^T
    cublasSgemmStridedBatched(
        CUBLAS_HANDLE,
        CUBLAS_OP_N, CUBLAS_OP_T,
        m,k,n,
        &alpha,
        out_grd.first<data::GPU>(), out_grd.ld, m,
        inp    .first<data::GPU>(), inp    .ld, k, &beta,
        wgt_grd.first<data::GPU>(), wgt_grd.ld, m,
        batches);

    // step 3. compute gradients of inputs
    // inp_grd = wgt^T * out_grd
    cublasSgemmStridedBatched(
        CUBLAS_HANDLE,
        CUBLAS_OP_T, CUBLAS_OP_N,
        k, n, m,
        &alpha,
        wgt    .first<data::GPU>(), wgt.ld    , m,
        out_grd.first<data::GPU>(), out_grd.ld, m, &beta,
        inp_grd.first<data::GPU>(), inp_grd.ld, k,
        batches);
}
}    // namespace operations

