#include "../../data/matrix_dense.h"
#include "../../data/matrix_sparse.h"

namespace operations {

// CUDA kernel for the backpropagation of element-wise multiplication
__global__ void elemwise_mul_bp_kernel(const float* __restrict__ inp1,
                                             float* __restrict__ inp1_grad,
                                       const float* __restrict__ inp2,
                                             float* __restrict__ inp2_grad,
                                       const float* __restrict__ out_grad,
                                       int rows,
                                       int cols,
                                       int ld) {

    // Calculate the row and column indices of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Calculate the indices using the MATRIX_INDEX macro
        int idx = MATRIX_INDEX(ld, row, col);

        // Compute the gradients
        if (inp1_grad != nullptr) {
            inp1_grad[idx] = inp2[idx] * out_grad[idx];
        }
        if (inp2_grad != nullptr) {
            inp2_grad[idx] = inp1[idx] * out_grad[idx];
        }
    }
}

// clang-format off
template<data::Device DEV>
inline void elemwise_mul_bp(const data::DenseMatrix<float>& inp1,
                                  data::DenseMatrix<float>& inp1_grad,
                            const data::DenseMatrix<float>& inp2,
                                  data::DenseMatrix<float>& inp2_grad,
                            const data::DenseMatrix<float>& out_grad) {
    // clang-format on

    ASSERT(inp1.m == inp2.m && inp1.m == out_grad.m);
    ASSERT(inp1.n == inp2.n && inp1.n == out_grad.n);
    ASSERT(inp1.ld == inp2.ld && inp1.ld == out_grad.ld);
    ASSERT(inp2.m == inp2_grad.m && inp2.n == inp2_grad.n && inp2.ld == inp2_grad.ld);

    if (data::is_gpu(DEV)) {
        // Define block and grid sizes
        dim3 blockSize(16, 16); // You can tune these values depending on your GPU architecture
        dim3 gridSize((inp1.n + blockSize.x - 1) / blockSize.x,
                      (inp1.m + blockSize.y - 1) / blockSize.y);

        // Call the CUDA kernel
        elemwise_mul_bp_kernel<<<gridSize, blockSize>>>(
            inp1     .first<data::GPU>(),
            inp1_grad.first<data::GPU>(),
            inp2     .first<data::GPU>(),
            inp2_grad.first<data::GPU>(),
            out_grad .first<data::GPU>(),
            inp1.m, inp1.n, inp1.ld);
        // Remember to check for kernel errors here as usual
    } else {
        // Use CPU implementation
        for (int i = 0; i < inp1.m; i++) {
            for (int j = 0; j < inp1.n; j++) {
                inp1_grad(i, j) = inp2(i, j) * out_grad(i, j);
                inp2_grad(i, j) = inp1(i, j) * out_grad(i, j);
            }
        }
    }
}

} // namespace operations
