#pragma once

#include "../../data/device.h"
#include "../../data/matrix_dense.h"
#include "../gradient_operation.h"

namespace operations {
// Use this macro to define both forward and backward operations for an activation function
#define DEFINE_ACTIVATION(activation_name, forward_expr, backward_expr)                              \
                                                                                                     \
    void activation_name##_host(const float* A,                                                      \
                                float*       B,                                                      \
                                float        scalar,                                                 \
                                size_t       m,                                                      \
                                size_t       n,                                                      \
                                size_t       lda,                                                    \
                                size_t       ldb) {                                                        \
        for (int x = 0; x < n; x++) {                                                                \
            for (int y = 0; y < m; y++) {                                                            \
                int ida = MATRIX_INDEX(lda, y, x);                                                   \
                int idb = MATRIX_INDEX(ldb, y, x);                                                   \
                B[idb]  = forward_expr;                                                              \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    __global__ void activation_name##_kernel(const float* __restrict__ A,                            \
                                             float* __restrict__ B,                                  \
                                             float  scalar,                                          \
                                             size_t m,                                               \
                                             size_t n,                                               \
                                             size_t lda,                                             \
                                             size_t ldb) {                                           \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                             \
        int idy = blockIdx.y * blockDim.y + threadIdx.y;                                             \
                                                                                                     \
        if (idx >= n || idy >= m)                                                                    \
            return;                                                                                  \
                                                                                                     \
        int ida = MATRIX_INDEX(lda, idy, idx);                                                       \
        int idb = MATRIX_INDEX(ldb, idy, idx);                                                       \
                                                                                                     \
        B[idb]  = forward_expr;                                                                      \
    }                                                                                                \
                                                                                                     \
    __global__ void activation_name##_kernel_fast(const float* __restrict__ A,                       \
                                                  float* __restrict__ B,                             \
                                                  float  scalar,                                     \
                                                  size_t size) {                                     \
        size_t ida = blockIdx.x * blockDim.x + threadIdx.x;                                          \
        size_t idb = ida;                                                                            \
        if (ida > size)                                                                              \
            return;                                                                                  \
        B[idb] = forward_expr;                                                                       \
    }                                                                                                \
                                                                                                     \
    void activation_name##_bp_host(const float*      A,                                              \
                                   float*            A_grd,                                          \
                                   const float*      B,                                              \
                                   const float*      B_grd,                                          \
                                                                                                     \
                                   float             scalar,                                         \
                                   size_t            m,                                              \
                                   size_t            n,                                              \
                                   size_t            lda,                                            \
                                   size_t            ldb,                                            \
                                   GradientOperation grad_operation) {                               \
        for (int x = 0; x < n; x++) {                                                                \
            for (int y = 0; y < m; y++) {                                                            \
                int ida    = MATRIX_INDEX(lda, y, x);                                                \
                int idb    = MATRIX_INDEX(ldb, y, x);                                                \
                A_grd[ida] = backward_expr + A_grd[ida] * grad_operation;                            \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    __global__ void activation_name##_bp_kernel(const float*      A,                                 \
                                                float*            A_grd,                             \
                                                const float*      B,                                 \
                                                const float*      B_grd,                             \
                                                                                                     \
                                                float             scalar,                            \
                                                size_t            m,                                 \
                                                size_t            n,                                 \
                                                size_t            lda,                               \
                                                size_t            ldb,                               \
                                                GradientOperation grad_operation) {                  \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                             \
        int idy = blockIdx.y * blockDim.y + threadIdx.y;                                             \
                                                                                                     \
        if (idx >= n || idy >= m)                                                                    \
            return;                                                                                  \
                                                                                                     \
        int ida    = MATRIX_INDEX(lda, idy, idx);                                                    \
        int idb    = MATRIX_INDEX(ldb, idy, idx);                                                    \
                                                                                                     \
        A_grd[ida] = backward_expr + A_grd[ida] * grad_operation;                                    \
    }                                                                                                \
                                                                                                     \
    __global__ void activation_name##_bp_kernel_fast(const float*      A,                            \
                                                     float*            A_grd,                        \
                                                     const float*      B,                            \
                                                     const float*      B_grd,                        \
                                                                                                     \
                                                     float             scalar,                       \
                                                     size_t            size,                         \
                                                     GradientOperation grad_operation) {             \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                             \
                                                                                                     \
        if (idx >= size)                                                                             \
            return;                                                                                  \
                                                                                                     \
        int ida    = idx;                                                                            \
        int idb    = ida;                                                                            \
                                                                                                     \
        A_grd[ida] = backward_expr + A_grd[ida] * grad_operation;                                    \
    }                                                                                                \
                                                                                                     \
    template<data::Device DEV>                                                                       \
    inline void activation_name(const data::DenseMatrix<float>& A,                                   \
                                data::DenseMatrix<float>&       B,                                   \
                                float                           scalar = 1) {                                                  \
                                                                                                     \
        ASSERT(A.size() == B.size());                                                                \
        ASSERT(A.first<DEV>());                                                                      \
        ASSERT(B.first<DEV>());                                                                      \
                                                                                                     \
        if constexpr (data::is_gpu(DEV)) {                                                           \
                                                                                                     \
            if (A.ld != A.m || B.ld != B.m) {                                                        \
                                                                                                     \
                int block_size_x;                                                                    \
                int block_size_y;                                                                    \
                if (A.m > 128) {                                                                     \
                    block_size_x = 1;                                                                \
                    block_size_y = 32;                                                               \
                } else if (A.m > 8) {                                                                \
                    block_size_x = 32;                                                               \
                    block_size_y = 8;                                                                \
                } else {                                                                             \
                    block_size_x = 512;                                                              \
                    block_size_y = 1;                                                                \
                };                                                                                   \
                                                                                                     \
                dim3 block(block_size_x, block_size_y);                                              \
                dim3 grid(std::ceil((float) A.n / block_size_x),                                     \
                          std::ceil((float) A.m / block_size_y));                                    \
                activation_name##_kernel<<<grid, block>>>(A.first<DEV>(),                            \
                                                          B.first<DEV>(),                            \
                                                          scalar,                                    \
                                                          A.m,                                       \
                                                          A.n,                                       \
                                                          A.ld,                                      \
                                                          B.ld);                                     \
            } else {                                                                                 \
                constexpr int block_size = 512;                                                      \
                                                                                                     \
                dim3          block(block_size);                                                     \
                dim3          grid(std::ceil((float) (A.m * A.n) / block_size));                     \
                activation_name##_kernel_fast<<<grid, block>>>(A.first<DEV>(),                       \
                                                               B.first<DEV>(),                       \
                                                               scalar,                               \
                                                               (A.m * A.n));                         \
            }                                                                                        \
        } else {                                                                                     \
            activation_name##_host(A.first<DEV>(), B.first<DEV>(), scalar, A.m, A.n, A.ld, B.ld);    \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    template<data::Device DEV>                                                                       \
    inline void activation_name##_bp(const data::DenseMatrix<float>& A,                              \
                                     data::DenseMatrix<float>&       A_grd,                          \
                                     const data::DenseMatrix<float>& B,                              \
                                     const data::DenseMatrix<float>& B_grd,                          \
                                     float                           scalar         = 1,             \
                                     GradientOperation               grad_operation = SET) {                       \
                                                                                                     \
        ASSERT(A.size() == B.size());                                                                \
        ASSERT(A.first<DEV>());                                                                      \
        ASSERT(A_grd.first<DEV>());                                                                  \
        ASSERT(B.first<DEV>());                                                                      \
        ASSERT(B_grd.first<DEV>());                                                                  \
                                                                                                     \
        if constexpr (data::is_gpu(DEV)) {                                                           \
                                                                                                     \
            if (A.ld != A.m || B.ld != B.m) {                                                        \
                                                                                                     \
                int block_size_x;                                                                    \
                int block_size_y;                                                                    \
                if (A.m > 128) {                                                                     \
                    block_size_x = 1;                                                                \
                    block_size_y = 32;                                                               \
                } else if (A.m > 8) {                                                                \
                    block_size_x = 32;                                                               \
                    block_size_y = 8;                                                                \
                } else {                                                                             \
                    block_size_x = 512;                                                              \
                    block_size_y = 1;                                                                \
                };                                                                                   \
                                                                                                     \
                dim3 block(block_size_x, block_size_y);                                              \
                dim3 grid(std::ceil((float) A.n / block_size_x),                                     \
                          std::ceil((float) A.m / block_size_y));                                    \
                activation_name##_bp_kernel<<<grid, block>>>(A.first<DEV>(),                         \
                                                             A_grd.first<DEV>(),                     \
                                                             B.first<DEV>(),                         \
                                                             B_grd.first<DEV>(),                     \
                                                             scalar,                                 \
                                                             A.m,                                    \
                                                             A.n,                                    \
                                                             A.ld,                                   \
                                                             B.ld,                                   \
                                                             grad_operation);                        \
            } else {                                                                                 \
                constexpr int block_size = 512;                                                      \
                                                                                                     \
                dim3          block(block_size);                                                     \
                dim3          grid(std::ceil((float) (A.m * A.n) / block_size));                     \
                activation_name##_bp_kernel_fast<<<grid, block>>>(A.first<DEV>(),                    \
                                                                  A_grd.first<DEV>(),                \
                                                                  B.first<DEV>(),                    \
                                                                  B_grd.first<DEV>(),                \
                                                                  scalar,                            \
                                                                  (A.m * A.n),                       \
                                                                  grad_operation);                   \
            }                                                                                        \
        } else {                                                                                     \
            activation_name##_bp_host(A.first<DEV>(),                                                \
                                      B.first<DEV>(),                                                \
                                      A_grd.first<DEV>(),                                            \
                                      B_grd.first<DEV>(),                                            \
                                      scalar,                                                        \
                                      A.m,                                                           \
                                      A.n,                                                           \
                                      A.ld,                                                          \
                                      B.ld,                                                          \
                                      grad_operation);                                               \
        }                                                                                            \
    }

}    // namespace operations