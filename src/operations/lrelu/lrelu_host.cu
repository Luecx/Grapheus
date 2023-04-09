
#include "lrelu.h"

#include <iostream>

// clang-format off
void operations::lrelu_host(
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

            if(A[ida] > 0){
                B[idb] = A[ida];
            }else{
                B[idb] = A[ida] / 16;
            }

        }
    }

}
