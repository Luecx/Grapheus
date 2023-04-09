
#include "relu.h"

#include <iostream>

// clang-format off
void operations::relu_host(
    const float* A,
          float* B,
    size_t m,
    size_t n,
    size_t lda,
    size_t ldb){
    // clang-format on

    for (int x = 0; x < n; x++){
        for(int y = 0; y < m; y++){
            int ida = MATRIX_INDEX(lda, y, x);
            int idb = MATRIX_INDEX(ldb, y, x);

            B[idb] = std::max(0.0f, A[ida]);
        }
    }

}
