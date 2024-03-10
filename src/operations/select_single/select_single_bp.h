//
// Created by finne on 07.04.2023.
//

#pragma once
#include "../../data/matrix_dense.h"
#include "../../data/sarray.h"
#include "../gradient_operation.h"

namespace operations {

// clang-format off
__global__ void select_single_bp_kernel(
          float* __restrict__ input,
    const float* __restrict__ output,
    const float* __restrict__ indices,
    size_t m,
    size_t n,
    size_t ldi,
    size_t ldo,
    size_t ldc,
    GradientOperation grad_op) {
    // clang-format on

    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_z = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_n >= n)
        return;

    int idx    = int(indices[MATRIX_INDEX(ldc, 0, thread_n)]);
    int offset = m * thread_z;

    if(grad_op == SET) {
        if (idx == thread_z) {
            for (int i = 0; i < m; i++) {
                input[MATRIX_INDEX(ldi, i + offset, thread_n)] = output[MATRIX_INDEX(ldo, i, thread_n)];
            }
        } else {
            for (int i = 0; i < m; i++) {
                input[MATRIX_INDEX(ldi, i + offset, thread_n)] = 0;
            }
        }
    } else {
        if (idx == thread_z) {
            for (int i = 0; i < m; i++) {
                input[MATRIX_INDEX(ldi, i + offset, thread_n)] += output[MATRIX_INDEX(ldo, i, thread_n)];
            }
        }
    }
}

// clang-format off
void select_single_bp_host(
          float* __restrict__ input,
          float* __restrict__ output,
    const float* __restrict__ indices,
    size_t m,
    size_t n,
    size_t heads,
    size_t ldi,
    size_t ldo,
    size_t ldc,
    GradientOperation grad_op) {
    // clang-format on
    for (int x = 0; x < n; x++) {
        size_t input_head = int(indices[MATRIX_INDEX(ldc, 0, x)]);
        size_t offset     = input_head * m;
        for (int z = 0; z < heads; z++) {
            for (int y = 0; y < m; y++) {
                if (input_head == z) {
                    input[MATRIX_INDEX(ldi, y + offset, x)] = output[MATRIX_INDEX(ldo, y, x)];
                } else {
                    input[MATRIX_INDEX(ldi, y + offset, x)] = 0;
                }
            }
        }
    }
}

// clang-format off
template<data::Device DEV>
void select_single_bp(      data::DenseMatrix<float>& input_grd,
                      const data::DenseMatrix<float>& output_grd,
                      const data::DenseMatrix<float>& indices,
                      GradientOperation grad_op = SET) {
    // clang-format on
    const size_t choices = input_grd.m / output_grd.m;

    // assert that memory is allocated for the output
    // cant really check that for the input pointers
    ASSERT(input_grd.address<DEV>());
    ASSERT(output_grd.address<DEV>());
    ASSERT(indices.address<DEV>());

    ASSERT(input_grd.n == output_grd.n);
    ASSERT(input_grd.n == indices.n);
    ASSERT(indices.m == 1);

    ASSERT(input_grd.m == output_grd.m * choices);

    if constexpr (data::is_gpu(DEV)) {
        // a block has 128 x 1 threads
        dim3 block(128, 1);
        // we need to spawn N threads in the y for N outputs
        dim3 grid(std::ceil((float) output_grd.n / 128.0f), choices);
        select_single_bp_kernel<<<grid, block>>>(input_grd.first<DEV>(),
                                                 output_grd.first<DEV>(),
                                                 indices.first<DEV>(),
                                                 output_grd.m,
                                                 output_grd.n,
                                                 input_grd.ld,
                                                 output_grd.ld,
                                                 indices.ld,
                                                 grad_op);
    } else {
        select_single_bp_host(input_grd.first<DEV>(),
                              output_grd.first<DEV>(),
                              indices.first<DEV>(),
                              output_grd.m,
                              output_grd.n,
                              choices,
                              input_grd.ld,
                              output_grd.ld,
                              indices.ld,
                              grad_op);
    }
}

}    // namespace operations
