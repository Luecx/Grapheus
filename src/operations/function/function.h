//
// Created by Luecx on 26.03.2023.
//

#pragma once
#include "../../data/device.h"
#include "../../data/matrix_dense.h"
#include "../../misc/start.h"
#include "../gradient_operation.h"

#include <cmath>

namespace math {

#define SHARED_SIMPLE_FUNCTION(name)                                                                 \
    template<typename TYPE>                                                                          \
    inline void name##_cpu(TYPE* inp, TYPE* out, size_t p_m, size_t p_n, size_t lda, size_t ldb) {   \
        for (size_t m = 0; m < p_m; m++) {                                                           \
            for (size_t n = 0; n < p_n; n++) {                                                       \
                out[MATRIX_INDEX(lda, m, n)] = std::name(inp[MATRIX_INDEX(ldb, m, n)]);              \
            }                                                                                        \
        }                                                                                            \
    }                                                                                                \
                                                                                                     \
    template<typename TYPE>                                                                          \
    __global__ void name##_gpu(TYPE*  inp,                                                           \
                               TYPE*  out,                                                           \
                               size_t p_m,                                                           \
                               size_t p_n,                                                           \
                               size_t lda,                                                           \
                               size_t ldb) {                                                         \
        int idx = blockIdx.x * blockDim.x + threadIdx.x;                                             \
        int idy = blockIdx.y * blockDim.y + threadIdx.y;                                             \
        if (idx >= p_m || idy >= p_n)                                                                \
            return;                                                                                  \
        out[MATRIX_INDEX(lda, idx, idy)] = std::name(inp[MATRIX_INDEX(ldb, idx, idy)]);              \
    }                                                                                                \
    template<data::Device DEVICE, typename TYPE>                                                     \
    inline void name(data::DenseMatrix<TYPE>* inp, data::DenseMatrix<TYPE>* out) {                   \
        ASSERT(inp->template is_allocated<DEVICE>());                                                \
        ASSERT(out->template is_allocated<DEVICE>());                                                \
        ASSERT(inp->m == out->m);                                                                    \
        ASSERT(inp->n == out->n);                                                                    \
        if constexpr (data::is_gpu(DEVICE)) {                                                        \
            constexpr int block_size = 32;                                                           \
            dim3          block(block_size, block_size);                                             \
            dim3          grid(std::ceil((float) inp->m / block_size),                               \
                      std::ceil((float) inp->n / block_size));                              \
            name##_gpu<<<grid, block>>>(inp->template first<data::GPU>(),                            \
                                        out->template first<data::GPU>(),                            \
                                        inp->m,                                                      \
                                        inp->n,                                                      \
                                        inp->ld,                                                     \
                                        out->ld);                                                    \
        }                                                                                            \
        if constexpr (data::is_cpu(DEVICE)) {                                                        \
            name##_cpu<TYPE>(inp->template first<data::CPU>(),                                       \
                             out->template first<data::CPU>(),                                       \
                             inp->m,                                                                 \
                             inp->n,                                                                 \
                             inp->ld,                                                                \
                             out->ld);                                                               \
        }                                                                                            \
    }

SHARED_SIMPLE_FUNCTION(sin);
SHARED_SIMPLE_FUNCTION(cos);
SHARED_SIMPLE_FUNCTION(tan);
SHARED_SIMPLE_FUNCTION(asin);
SHARED_SIMPLE_FUNCTION(acos);
SHARED_SIMPLE_FUNCTION(atan)
SHARED_SIMPLE_FUNCTION(atan2);

SHARED_SIMPLE_FUNCTION(sinh);
SHARED_SIMPLE_FUNCTION(cosh);
SHARED_SIMPLE_FUNCTION(tanh);
SHARED_SIMPLE_FUNCTION(asinh);
SHARED_SIMPLE_FUNCTION(acosh);
SHARED_SIMPLE_FUNCTION(atanh);

SHARED_SIMPLE_FUNCTION(erf);
SHARED_SIMPLE_FUNCTION(erfc);

SHARED_SIMPLE_FUNCTION(ceil);
SHARED_SIMPLE_FUNCTION(floor);
SHARED_SIMPLE_FUNCTION(trunc)
SHARED_SIMPLE_FUNCTION(round);

SHARED_SIMPLE_FUNCTION(exp);
SHARED_SIMPLE_FUNCTION(exp2);
SHARED_SIMPLE_FUNCTION(log)
SHARED_SIMPLE_FUNCTION(log10);
SHARED_SIMPLE_FUNCTION(log2);
SHARED_SIMPLE_FUNCTION(sqrt);
SHARED_SIMPLE_FUNCTION(cbrt);

SHARED_SIMPLE_FUNCTION(abs);
SHARED_SIMPLE_FUNCTION(fmod);
SHARED_SIMPLE_FUNCTION(remainder);

}    // namespace math
#undef SHARED_SIMPLE_FUNCTION

