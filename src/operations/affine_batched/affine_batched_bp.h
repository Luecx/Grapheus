//
// Created by finne on 09.04.2023.
//

#pragma once
namespace operations {

/**
 * Performs backpropagation through a batched affine transformation on the GPU. This operation
 * computes the gradient of the loss with respect to the inputs, weights, and biases of the affine
 * transformation.
 *
 * Given the input matrices `x_i` and their corresponding weight matrices `W_i` and bias vectors
 * `b_i`, the forward pass computes:
 *
 *     y_i = W_i * x_i + b_i
 *
 * For multiple inputs and weights using the same bias vector. The backward pass computes the
 * gradients of the loss with respect to each of these matrices.
 *
 * The inputs, weights, biases, and their gradients are represented as arrays of pointers to the first
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
 * Note that the function expects the output gradient pointers to be sub-matrices of a larger output
 * gradient matrix. Specifically, if there are `num_batches` batches and the output size is `m` by
 * `n`, then each output gradient matrix must be a sub-matrix of a consecutive block of the output
 * gradient matrix of size `num_batches * m` by `n`. Visually, this can be interpreted as:
 *
 *  |<----- n ---->|
 *  |              |
 *  +--------------+
 *  |    batch 0   |  <-- out_grd[0]
 *  +--------------+
 *  |    batch 1   |  <-- out_grd[1]
 *  +--------------+
 *  |    batch 2   |  <-- out_grd[2]
 *  +--------------+
 *  |      ...     |   ...
 *  +--------------+
 *  | batch count-1|
 *  +--------------+
 *
 * Similarly, each input gradient matrix, weight gradient matrix, and bias gradient vector must be a
 * sub-matrix or sub-vector of a consecutive block of the corresponding matrix or vector of size
 * `num_batches * k` by `n`, `num_batches * m` by `k`, and `num_batches * m` by 1, respectively.
 *
 * @tparam DEV the device type, only works with data::GPU.
 * @param inp array of pointers to input matrices of shape `inp = [k, n]`.
 * @param inp_grd array of pointers to input gradient matrices of shape `inp_grd = [k, n]`.
 * @param wgt array of pointers to weight matrices of shape `wgt = [m, k]`.
 * @param wgt_grd array of pointers to weight gradient matrices of shape `wgt_grd = [m, k]`.
 * @param bia_grd array of pointers to bias gradient vectors of shape `bia_grd = [m, 1]`.
 * @param out_grd array of pointers to output gradient matrices of shape `out_grd = [m, n]`.
 * @param m the number of rows in the output and bias matrices.
 * @param n the number of columns in the input and output matrices.
 * @param k the number of columns in the weight matrix.
 * @param ld_inp the leading dimension of the input / input-gradient matrices.
 * @param ld_wgt the leading dimension of the weight / weight-gradient matrices.
 * @param ld_out the leading dimension of the output-gradient matrices.
 */
// clang-format off
template<data::Device DEV>
inline void affine_batched_bp(const data::SArray<float*>& inp,
                                    data::SArray<float*>& inp_grd,
                              const data::SArray<float*>& wgt,
                                    data::SArray<float*>& wgt_grd,
                                    data::SArray<float*>& bia_grd,
                              const data::SArray<float*>& out_grd,
                              const size_t m,
                              const size_t n,
                              const size_t k,
                              const size_t ld_inp,
                              const size_t ld_wgt,
                              const size_t ld_out) {
    // make sure we can access the pointers at least
    ASSERT(inp    .is_allocated<DEV>());
    ASSERT(wgt    .is_allocated<DEV>());
    ASSERT(wgt_grd.is_allocated<DEV>());
    ASSERT(bia_grd.is_allocated<DEV>());
    ASSERT(out_grd.is_allocated<DEV>());
    ASSERT(inp_grd.is_allocated<DEV>());

    // also make sure we have the same amount of inputs, weights, biases and outputs
    ASSERT(inp    .size() == inp_grd.size());
    ASSERT(inp_grd.size() == wgt    .size());
    ASSERT(wgt    .size() == wgt_grd.size());
    ASSERT(wgt_grd.size() == bia_grd.size());
    ASSERT(bia_grd.size() == out_grd.size());
    // clang-format on

    // make sure we have no invalid leading dimensions
    ASSERT(ld_inp && ld_wgt && ld_out);

    // make sure the pointers for the outputs and biases are evenly spaces
    ASSERT((out_grd[1] - out_grd[0]) == m);
    ASSERT((bia_grd[1] - bia_grd[0]) == m);

    const size_t batch_count = inp.size();

    // step 1: reduce rows to get the gradient of bias
    constexpr int block_size_x = 128;
    constexpr int block_size_y = 8;
    dim3 block(block_size_x, block_size_y);
    dim3 grid (std::ceil((float)n               / block_size_x / 2),
               std::ceil((float)m * batch_count / block_size_y));
    cudaMemset(bia_grd[0], 0, sizeof(float) * m * batch_count);
    reduce_row<<<grid, block>>>(
        out_grd[0],
        bia_grd[0],
        m * batch_count,
        n,
        ld_out);

    const float alpha = 1;
    const float beta  = 0;
    // step 2. compute gradients of weights
    // wgt_grd = inp * out_grd^T
    cublasSgemmBatched(CUBLAS_HANDLE,
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       m,k,n,
                       &alpha,
                       out_grd.address<data::GPU>(), ld_out,
                       inp    .address<data::GPU>(), ld_inp, &beta,
                       wgt_grd.address<data::GPU>(), ld_wgt,
                       batch_count);

    // step 3. compute gradients of inputs
    // inp_grd = wgt^T * out_grd
    cublasSgemmBatched(CUBLAS_HANDLE,
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       k, n, m,
                       &alpha,
                       wgt    .address<data::GPU>(), ld_wgt,
                       out_grd.address<data::GPU>(), ld_out, &beta,
                       inp_grd.address<data::GPU>(), ld_inp,
                       batch_count);
}
}    // namespace operations

