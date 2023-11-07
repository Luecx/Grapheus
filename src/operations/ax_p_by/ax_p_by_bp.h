#pragma once

#include "../../data/sarray.h"
#include "../../data/device.h"
#include "../../data/matrix_dense.h"
#include "../gradient_operation.h"

namespace operations {

// clang-format off
template<data::Device DEV>
inline void ax_p_by_bp(      data::DenseMatrix<float>& inp1,
                             data::DenseMatrix<float>& inp2,
                       const data::DenseMatrix<float>& out,
                       float inp1_scalar,
                       float inp2_scalar,
                       GradientOperation grad_operation = SET) {

    // clang-format on
    ASSERT(inp1.m == inp2.m && inp1.m == out.m);
    ASSERT(inp1.n == inp2.n && inp1.n == out.n);

    if (data::is_gpu(DEV)){
         const float zero = grad_operation == SET ? 0:1;
         cublasSgeam(CUBLAS_HANDLE, CUBLAS_OP_N, CUBLAS_OP_N,
                    inp1.m, inp1.n,
                    &inp1_scalar, out .first<DEV>(), out. ld,
                    &zero       , inp1.first<DEV>(), inp1.ld,
                                  inp1.first<DEV>(), inp1.ld);
         cublasSgeam(CUBLAS_HANDLE, CUBLAS_OP_N, CUBLAS_OP_N,
                    inp1.m, inp1.n,
                    &inp2_scalar, out .first<DEV>(), out. ld,
                    &zero       , inp2.first<DEV>(), inp2.ld,
                                  inp2.first<DEV>(), inp2.ld);
    }else{
        // Use CPU implementation
        for (int i = 0; i < inp1.m; i++) {
            for (int j = 0; j < inp1.n; j++) {
                inp1(i, j) = inp1_scalar * out(i, j) + grad_operation * inp1(i, j);
                inp2(i, j) = inp2_scalar * out(i, j) + grad_operation * inp2(i, j);
            }
        }
    }
}

}    // namespace operations
