//
// Created by finne on 07.04.2023.
//

#pragma once

#include "../../data/matrix_dense.h"

namespace operations {

// clang-format off
__global__ void select_kernel(
          float** __restrict__ inputs,
          float*  __restrict__ output,
    const float*  __restrict__ indices,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on

    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_n >= n)
        return;

    int          idx   = indices[thread_n];
    const float* input = inputs[idx];

    for (int i = 0; i < m; i++) {
        output[MATRIX_INDEX(ld, i, thread_n)] = input[MATRIX_INDEX(ld, i, thread_n)];
    }
}

// clang-format off
void select_host(
          float** __restrict__ inputs,
          float*  __restrict__ output,
    const float*  __restrict__ indices,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on
    for (int x = 0; x < n; x++) {
        int in_head = int(indices[x]);
        for (int y = 0; y < m; y++) {
            output[MATRIX_INDEX(ld, x, y)] = inputs[in_head][MATRIX_INDEX(ld, x, y)];
        }
    }
};

// clang-format off
template<data::Device DEV>
void select(      data::SArray     <float*>& inputs,
                  data::DenseMatrix<float> & output,
            const data::SArray     <float >& indices) {
    // clang-format on
    // assert that memory is allocated for the output
    // cant really check that for the input pointers
    ASSERT(inputs .address<DEV>());
    ASSERT(output .address<DEV>());
    ASSERT(indices.address<DEV>());

    if constexpr (data::is_gpu(DEV)) {
        // a block has 128 threads in n-direction
        dim3 block(128);
        // we need to spawn N threads in the y for N outputs
        // this equals N / 128 blocks, each with 128 threads
        // each thread takes care of all the values in the m direction
        dim3 grid(std::ceil((float) output.n / 128.0f));
        select_kernel<<<grid, block>>>(inputs.address<DEV>(),
                                       output.first<DEV>(),
                                       indices.address<DEV>(),
                                       output.m,
                                       output.n,
                                       output.ld);
    } else {
        select_host(inputs.address<DEV>(),
                    output.first<DEV>(),
                    indices.address<DEV>(),
                    output.m,
                    output.n,
                    output.ld);
    }
}

}    // namespace operations

