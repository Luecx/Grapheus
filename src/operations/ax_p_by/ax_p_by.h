#pragma once

namespace operations {

// clang-format off
template<data::Device DEV>
inline void ax_p_by(const data::DenseMatrix<float>& inp1,
                    const data::DenseMatrix<float>& inp2,
                          data::DenseMatrix<float>& out,
                    float inp1_scalar,
                    float inp2_scalar) {
    // clang-format on

    ASSERT(inp1.m == inp2.m && inp1.m == out.m);
    ASSERT(inp1.n == inp2.n && inp1.n == out.n);

    if (data::is_gpu(DEV)){
        cublasSgeam(CUBLAS_HANDLE, CUBLAS_OP_N, CUBLAS_OP_N,
                    inp1.m, inp1.n,
                    &inp1_scalar, inp1.first<DEV>(), inp1.ld,
                    &inp2_scalar, inp2.first<DEV>(), inp2.ld,
                                  out .first<DEV>(), out. ld);
    }else{
        // Use CPU implementation
        for (int i = 0; i < inp1.m; i++) {
            for (int j = 0; j < inp1.n; j++) {
                out(i, j) = inp1_scalar * inp1(i, j) + inp2_scalar * inp2(i, j);
            }
        }
    }
}

}    // namespace operations
