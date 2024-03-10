#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"

namespace operations {

// CUDA kernel for element-wise multiplication of two matrices
__global__ void elemwise_mul_kernel(const float* __restrict__ inp1,
                                    const float* __restrict__ inp2,
                                          float* __restrict__ out,
                                    int rows,
                                    int cols,
                                    int ld_inp1,
                                    int ld_inp2,
                                    int ld_out) {

    // Calculate the row index of the element and check if within bounds
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the element and check if within bounds
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols) {
        // Calculate the index using the provided MATRIX_INDEX macro
        int idx_inp1 = MATRIX_INDEX(ld_inp1, row, col);
        int idx_inp2 = MATRIX_INDEX(ld_inp2, row, col);
        int idx_out  = MATRIX_INDEX(ld_out , row, col);

        // Perform the multiplication
        out[idx_out] = inp1[idx_inp1] * inp2[idx_inp2];
    }
}


// clang-format off
template<data::Device DEV>
inline void elemwise_mul(const data::DenseMatrix<float>& inp1,
                         const data::DenseMatrix<float>& inp2,
                               data::DenseMatrix<float>& out) {
    // clang-format on

    ASSERT(inp1.m == inp2.m && inp1.m == out.m);
    ASSERT(inp1.n == inp2.n && inp1.n == out.n);

    if (data::is_gpu(DEV)){
        // Define block and grid sizes
        dim3 blockSize(16, 16); // You can tune these values depending on your GPU architecture
        dim3 gridSize((inp1.n + blockSize.x - 1) / blockSize.x,
                      (inp1.m + blockSize.y - 1) / blockSize.y);

        // Call the CUDA kernel
        elemwise_mul_kernel<<<gridSize, blockSize>>>(
            inp1.first<data::GPU>(),
            inp2.first<data::GPU>(),
            out .first<data::GPU>(),
            inp1.m, inp1.n, inp1.ld, inp2.ld, out.ld);
    }else{
        // Use CPU implementation
        for (int i = 0; i < inp1.m; i++) {
            for (int j = 0; j < inp1.n; j++) {
                out(i, j) = inp1(i, j) * inp2(i, j);
            }
        }
    }
}

}    // namespace operations
