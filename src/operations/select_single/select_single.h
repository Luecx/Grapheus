//
// Created by finne on 07.04.2023.
//

#pragma once
namespace operations {

// clang-format off
__global__ void select_single_kernel(
          float* __restrict__ input,
          float* __restrict__ output,
    const float* __restrict__ indices,
    size_t m,
    size_t n,
    size_t ldi,
    size_t ldo,
    size_t ldc) {
    // clang-format on

    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_n >= n)
        return;

    int idx    = indices[MATRIX_INDEX(ldc, 0, thread_n)];
    int offset = m * idx;

    for (int i = 0; i < m; i++) {
        output[MATRIX_INDEX(ldo, i, thread_n)] = input[MATRIX_INDEX(ldi, i + offset, thread_n)];
    }
}

// clang-format off
void select_single_host(
          float* __restrict__ input,
          float* __restrict__ output,
    const float* __restrict__ indices,
    size_t m,
    size_t n,
    size_t ldi,
    size_t ldo,
    size_t ldc) {

    // clang-format on
    for (int x = 0; x < n; x++) {
        int in_head = int(indices[MATRIX_INDEX(ldc, 0, x)]);
        int offset = m * in_head;
        for (int y = 0; y < m; y++) {
            output[MATRIX_INDEX(ldo, y, x)] = input[MATRIX_INDEX(ldi, y + offset, x)];
        }
    }
};

// clang-format off
template<data::Device DEV>
void select_single(const data::DenseMatrix<float>& input,
                         data::DenseMatrix<float>& output,
                   const data::DenseMatrix<float>& indices) {

    // clang-format on
    [[maybe_unused]] const size_t choices = input.m / output.m;

    // assert that memory is allocated for the output
    // cant really check that for the input pointers
    ASSERT(input.address<DEV>());
    ASSERT(output.address<DEV>());
    ASSERT(indices.address<DEV>());

    ASSERT(input.n == output.n);
    ASSERT(input.n == indices.n);
    ASSERT(indices.m == 1);

    ASSERT(input.m == output.m * choices);

    if constexpr (data::is_gpu(DEV)) {
        // a block has 128 threads in n-direction
        dim3 block(128);
        // we need to spawn N threads in the n-direction
        // this equals N / 128 blocks, each with 128 threads
        // each thread takes care of all the values in the m direction
        dim3 grid(std::ceil((float) output.n / 128.0f));
        select_single_kernel<<<grid, block>>>(input.first<DEV>(),
                                              output.first<DEV>(),
                                              indices.first<DEV>(),
                                              output.m,
                                              output.n,
                                              input.ld,
                                              output.ld,
                                              indices.ld);
    } else {
        select_single_host(input.first<DEV>(),
                           output.first<DEV>(),
                           indices.first<DEV>(),
                           output.m,
                           output.n,
                           input.ld,
                           output.ld,
                           indices.ld);
    }
}

}    // namespace operations
