//
// Created by finne on 07.04.2023.
//

#pragma once
namespace operations {

// clang-format off
__global__ void distribute_kernel(
    const float*  __restrict__ input,
          float** __restrict__ outputs,
    const float*  __restrict__ indices,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on

    int thread_n = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_z = blockIdx.y * blockDim.y + threadIdx.y;

    if (thread_n >= n)
        return;

    float* output =     outputs[thread_z];
    int    idx    = int(indices[thread_n]);

    if (idx == thread_z) {
        for (int i = 0; i < m; i++) {
            output[MATRIX_INDEX(ld, i, thread_n)] = input[MATRIX_INDEX(ld, i, thread_n)];
        }
    } else {
        // TODO: reconsider
        // NOTE: technically correct but in reality not needed
        for (int i = 0; i < m; i++) {
            output[MATRIX_INDEX(ld, i, thread_n)] = 0;
        }
    }
}

// clang-format off
void distribute_host(
          float*  __restrict__ input,
          float** __restrict__ outputs,
    const float*  __restrict__ indices,
    unsigned int heads,
    unsigned int m,
    unsigned int n,
    unsigned int ld) {
    // clang-format on
    for (int x = 0; x < n; x++) {
        size_t output_head = int(indices[x]);
        for(int z = 0; z < heads; z++){
            float* output = outputs[z];
            for(int y = 0; y < m; y++){
                if(output_head == z){
                    output[MATRIX_INDEX(ld,y,x)] = input[MATRIX_INDEX(ld,y,x)];
                }else{
                    output[MATRIX_INDEX(ld,y,x)] = 0;
                }
            }
        }
    }
}

/**
 * performs a distribution task from a single batched input to multiple outputs depending
 * on an index array.
 * To be precise, the input is assumed to be a MxN matrix where N is usually the batch-size
 * in neural network applications. Together with the input matrix, there are indices given
 * which describe for each column of the input matrix, to which output matrix it will be put.
 *
 * --------------------------------------------------------------------------------------------------
 * An example would be:
 *
 * input matrix: [2 x 4]
 *     0.81472    0.13548    0.90579    0.83501
 *     0.12699    0.96887    0.91338    0.22103
 *
 * indices for each column:
 *     0          0          1          0
 *
 *
 * outputs:
 *
 * output 0:
 * 0.81472    0.13548    0.00000    0.83501
 * 0.12699    0.96887    0.00000    0.22103
 *
 * output 1:
 * 0.00000    0.00000    0.90579    0.00000
 * 0.00000    0.00000    0.91338    0.00000
 *
 * @tparam DEV
 * @param input
 * @param outputs
 * @param indices
 */
// clang-format off
template<data::Device DEV>
void distribute(const data::DenseMatrix<float> & input,
                      data::SArray     <float*>& outputs,
                const data::SArray     <float >& indices) {
    // clang-format on

    // not possible here to check for each individual output pointer
    ASSERT(input.address<DEV>());
    ASSERT(outputs.address<DEV>());
    ASSERT(indices.address<DEV>());

    if constexpr (data::is_gpu(DEV)) {
        // a block has 128 x 1 threads
        dim3 block(128, 1);
        // we need to spawn N threads in the y for N outputs
        dim3 grid(std::ceil((float) input.n / 128.0f), outputs.size());
        distribute_kernel<<<grid, block>>>(input.first<DEV>(),
                                           outputs.address<DEV>(),
                                           indices.address<DEV>(),
                                           input.m,
                                           input.n,
                                           input.ld);
    } else {
        distribute_host(input.first<DEV>(),
                        outputs.address<DEV>(),
                        indices.address<DEV>(),
                        outputs.size(),
                        input.m,
                        input.n,
                        input.ld);
    }
}

}    // namespace operations

