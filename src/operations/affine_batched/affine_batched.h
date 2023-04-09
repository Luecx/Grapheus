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
 * The inputs, weights, biases, and outputs are represented as arrays of pointers to the first
 * elements of each matrix. It is assumed that the memory for all matrices has already been allocated
 * and is accessible on the GPU.
 *
 * The dimensions of the matrices are:
 *
 *     out = [m, n]
 *     bia = [m, 1]
 *     inp = [k, n]
 *     wgt = [m, k]
 *
 * The indexing into each array is done using the leading dimension. The function assumes that all
 * inputs have the same size.
 *
 * Note that the function expects the output pointers to be sub-matrices of a larger output matrix,
 * and similarly for the bias vectors. Specifically, if there are `num_batches` batches and the output
 * size is `m` by `n`, then each output matrix must be a sub-matrix of a consecutive block of the
 * output matrix of size `num_batches * m` by `n`. Similarly, each bias vector must be a sub-vector of
 * a consecutive block of the bias matrix of size `num_batches * m` by 1. Visually, this can be
 * interpreted as:
 *
 *  |<----- n ---->|
 *  |              |
 *  +--------------+
 *  |    batch 0   |  <-- out[0]
 *  +--------------+
 *  |    batch 1   |  <-- out[1]
 *  +--------------+
 *  |    batch 2   |  <-- out[2]
 *  +--------------+
 *  |      ...     |   ...
 *  +--------------+
 *  | batch count-1|
 *  +--------------+
 *
 *  Also note that the batch count in the context of neural networks is not the same. In the context
 *  of neural networks, the batch size would be the width of the output / input matrix [n]
 *
 *
 * @tparam DEV the device type, only works with data::GPU
 * @param inp array of pointers to input matrices of shape `inp = [k, n]`.
 * @param wgt array of pointers to weight matrices of shape `wgt = [m, k]`.
 * @param bia array of pointers to bias vectors of shape `bia = [m, 1]`.
 * @param out array of pointers to output matrices of shape `out = [m, n]`.
 * @param m the number of rows in the output and bias matrices.
 * @param n the number of columns in the input and output matrices.
 * @param k the number of columns in the weight matrix.
 * @param ld_inp the leading dimension of the input matrices.
 * @param ld_wgt the leading dimension of the weight matrices.
 * @param ld_out the leading dimension of the output matrices.
 */
// clang-format off
template<data::Device DEV>
inline void affine_batched(const data::SArray<float*>& inp,
                           const data::SArray<float*>& wgt,
                           const data::SArray<float*>& bia,
                                 data::SArray<float*>& out,
                           const size_t m,
                           const size_t n,
                           const size_t k,
                           const size_t ld_inp,
                           const size_t ld_wgt,
                           const size_t ld_out) {

    // clang-format on

    // make sure we can access the pointers at least
    ASSERT(inp.is_allocated<DEV>());
    ASSERT(wgt.is_allocated<DEV>());
    ASSERT(bia.is_allocated<DEV>());
    ASSERT(out.is_allocated<DEV>());

    // also make sure we have the same amount of inputs, weights, biases and outputs
    ASSERT(inp.size() == wgt.size());
    ASSERT(wgt.size() == bia.size());
    ASSERT(bia.size() == out.size());

    // make sure we have no invalid leading dimensions
    ASSERT(ld_inp && ld_wgt && ld_out);

    // make sure the pointers for the outputs and biases are evenly spaces
    ASSERT((out[1] - out[0]) == m);
    ASSERT((bia[1] - bia[0]) == m);


    const size_t batch_count = inp.size();
    const float  alpha       = 1;
    const float  beta        = 0;

    // step 1: compute the matrix multiplication for every batch
    cublasSgemmBatched(CUBLAS_HANDLE,
                       CUBLAS_OP_N,CUBLAS_OP_N,
                       m,n,k,
                       &alpha,
                       wgt.address<data::GPU>(), ld_wgt,
                       inp.address<data::GPU>(), ld_inp,
                       &beta,
                       out.address<data::GPU>(), ld_out,
                       batch_count);

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
               std::ceil((float)m * batch_count / block_size_y));

    add_bias_kernel<<<grid, block>>>(bia[0],
                                     out[0],
                                     m * batch_count,
                                     n,
                                     ld_out);
}

}    // namespace operations
