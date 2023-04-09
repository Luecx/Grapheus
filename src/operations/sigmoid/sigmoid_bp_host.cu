
#include "sigmoid_bp.h"

#include <iostream>

// clang-format off
void operations::sigmoid_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    float scalar,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb){
    // clang-format on

    for (int x = 0; x < n; x++){
        for(int y = 0; y < m; y++){
            int ida = MATRIX_INDEX(lda, y, x);
            int idb = MATRIX_INDEX(ldb, y, x);
            A_grd[ida] = B_grd[idb] * B[idb] * (1 - B[idb]) * scalar;
        }
    }

}
