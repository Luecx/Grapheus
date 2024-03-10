//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include <cublas_v2.h>
#include <cusparse_v2.h>

extern cublasHandle_t   CUBLAS_HANDLE;
extern cusparseHandle_t CUSPARSE_HANDLE;

void                    init();
void                    close();
