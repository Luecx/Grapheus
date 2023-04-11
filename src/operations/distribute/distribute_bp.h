////
//// Created by finne on 07.04.2023.
////
//
//#pragma once
//#include "../../data/matrix_dense.h"
//#include "../../data/sarray.h"
//
//namespace operations {
//
//// clang-format off
//__global__ void distribute_bp_kernel(
//          float*  __restrict__ input,
//          float** __restrict__ outputs,
//    const float * __restrict__ indices,
//    unsigned int m,
//    unsigned int n,
//    unsigned int ld) {
//    // clang-format on
//
//    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;
//    int thread_z = blockIdx.y * blockDim.y + threadIdx.y;
//
//    if (thread_n >= n)
//        return;
//
//    const float* output = outputs[thread_z];
//    int          idx    = indices[thread_n];
//
//    if (idx == thread_z) {
//        for (int i = 0; i < m; i++) {
//            input[MATRIX_INDEX(ld, i, thread_n)] = output[MATRIX_INDEX(ld, i, thread_n)];
//        }
//    }
//}
//
//// clang-format off
//void distribute_bp_host(
//          float*  __restrict__ input,
//          float** __restrict__ outputs,
//    const float * __restrict__ indices,
//    unsigned int m,
//    unsigned int n,
//    unsigned int ld) {
//    // clang-format on
//
//    for (int x = 0; x < n; x++) {
//        int out_head = int(indices[x]);
//        for (int y = 0; y < m; y++) {
//            input[MATRIX_INDEX(ld, x, y)] = outputs[out_head][MATRIX_INDEX(ld, x, y)];
//        }
//    }
//}
//
///**
// * backpropagation through the selection process.
// * The forward pass is defined as:
// * The input is assumed to be a MxN matrix where N is usually the batch-size
// * in neural network applications. Together with the input matrix, there are indices given
// * which describe for each column of the input matrix, in which output matrix it will be put.
// *
// * @tparam DEV
// * @param input
// * @param outputs
// * @param indices
// */
//// clang-format off
//template<data::Device DEV>
//void distribute_bp(      data::DenseMatrix<float> & input,
//                   const data::SArray     <float*>& outputs,
//                   const data::SArray     <float >& indices) {
//    // clang-format on
//    if constexpr (data::is_cpu(DEV)) {
//        // a block has 128 x 1 threads
//        dim3 block(128, 1);
//        // we need to spawn N threads in the y for N outputs
//        dim3 grid(std::ceil((float) input.n / 128.0f), outputs.size());
//        distribute_bp_kernel<<<grid, block>>>(input.first<DEV>(),
//                                              outputs.address<DEV>(),
//                                              indices.address<DEV>(),
//                                              input.m,
//                                              input.n,
//                                              input.ld);
//    } else {
//        distribute_bp_host(input.first<DEV>(),
//                           outputs.address<DEV>(),
//                           indices.address<DEV>(),
//                           input.m,
//                           input.n,
//                           input.ld);
//    }
//}
//
//}    // namespace operations
//
