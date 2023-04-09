
#include "relu_bp.h"

#include <iostream>

// clang-format off
void operations::relu_bp_host(
    const float* A,
          float* A_grd,
    const float* B,
    const float* B_grd,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb){
    // clang-format on

    for (int x = 0; x < n; x++){
        for(int y = 0; y < m; y++){
            int ida = MATRIX_INDEX(lda, y, x);
            int idb = MATRIX_INDEX(ldb, y, x);

            if (B[idb] > 0) {
                A_grd[ida] = B_grd[idb];
            } else {
                A_grd[ida] = 0;
            }
        }
    }

}
