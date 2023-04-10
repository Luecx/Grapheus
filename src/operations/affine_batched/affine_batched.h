//
// Created by finne on 09.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"
#include "../affine/affine.h"

namespace operations{
/**
 * Performs a batched affine transformation on the GPU. This operation computes the matrix product
 *
 *     y_i = W_i * x_i + b_i
 *
 * for multiple inputs `x_i` using different weight matrices `W_i` and bias vectors `b_i`. This is
 * more efficient than computing each affine transformation separately.
 *
 * The inputs, weights, biases, and outputs are represented as sub-matrices of a larger matrix
 * for each field respectively.
 *
 * The dimensions of the matrices are:
 *
 *     out = [B * m, n]
 *     bia = [B * m, 1]
 *     inp = [B * k, n]
 *     wgt = [B * m, k]
 *
 * where B is the amount of affine transformations to be performed. The function takes in the
 * matrices out, bia, inp and wgt and extract the submatrices. It is assumed that the submatrices
 * are evenly spaced.
 *
 * The indexing into each array is done using the leading dimension. The function assumes that all
 * inputs have the same size.
 *
 *
 *  |<----- n ---->|
 *  |              |
 *  +--------------+
 *  |    sub 0     |  <-- out[0]
 *  +--------------+
 *  |    sub 1     |  <-- out[1]
 *  +--------------+
 *  |    sub 2     |  <-- out[2]
 *  +--------------+
 *  |      ...     |   ...
 *  +--------------+
 *  |    sub B-1   |  <-- out[B-1]
 *  +--------------+
 *
 *  Also note that the batch count in the context of neural networks is not the same. In the context
 *  of neural networks, the batch size would be the width of the output / input matrix [n]
 *
 *
 * @tparam DEV the device type, only works with data::GPU
 * @param inp array of pointers to input matrices of shape `inp = [B * k, n]`.
 * @param wgt array of pointers to weight matrices of shape `wgt = [B * m, k]`.
 * @param bia array of pointers to bias vectors of shape `bia = [B * m, 1]`.
 * @param out array of pointers to output matrices of shape `out = [B * m, n]`.
 * @param batches amount of batches to perform. Also the amount of sub-matrices per matrix
 */
// clang-format off
template<data::Device DEV>
inline void affine_batched(const data::DenseMatrix<float>& inp,
                           const data::DenseMatrix<float>& wgt,
                           const data::DenseMatrix<float>& bia,
                                 data::DenseMatrix<float>& out,
                           const size_t batches) {

    // clang-format on

    // make sure we can access the pointers at least
    ASSERT(inp.is_allocated<DEV>());
    ASSERT(wgt.is_allocated<DEV>());
    ASSERT(bia.is_allocated<DEV>());
    ASSERT(out.is_allocated<DEV>());

    // also make sure we have the same amount of inputs, weights, biases and outputs
    ASSERT(inp.m == wgt.n * batches);
    ASSERT(out.m == wgt.m);
    ASSERT(out.m == bia.m);
    ASSERT(bia.n == 1);
    ASSERT(inp.n == out.n);

    // make sure we have no invalid leading dimensions
    ASSERT(inp.ld && wgt.ld && out.ld);

    // make sure the size is actually divisible by the amount of batches
    ASSERT(inp.m % batches == 0);
    ASSERT(wgt.m % batches == 0);
    ASSERT(bia.m % batches == 0);
    ASSERT(out.m % batches == 0);

    const size_t m = out.m / batches;
    const size_t n = out.n;
    const size_t k = wgt.n;

    const float  alpha       = 1;
    const float  beta        = 0;

    // step 1: compute the matrix multiplication for every batch
    cublasSgemmStridedBatched(CUBLAS_HANDLE,
                       CUBLAS_OP_N,CUBLAS_OP_N,
                       m,n,k,
                       &alpha,
                       wgt.first<DEV>(), wgt.ld, m,
                       inp.first<DEV>(), inp.ld, k,
                       &beta,
                       out.first<DEV>(), out.ld, m,
                       batches);

    // step 2: add the bias to the output
    // trick here is that we assume all biases and outputs
    // are part of a bigger submatrix. Since we checked that before inside the ASSERT,
    // we can simply call a single bias-addition to the output matrix. For this we simply
    // need to give the pointers to the first bias / output element and multiply the height (m)
    // by the batch_count. This saves operations
    constexpr int block_size_x = 2;
    constexpr int block_size_y = 32;
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)n               / block_size_x),
               std::ceil((float)m * batches     / block_size_y));

    add_bias_kernel<<<grid, block>>>(bia.first<DEV>(),
                                     out.first<DEV>(),
                                     bia.m,
                                     n,
                                     out.ld);
}

}    // namespace operations
