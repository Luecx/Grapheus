#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/sarray.h"
#include "../layers/tape.h"

namespace nn {

struct Loss {

    data::SArray<float> loss {1};
    data::DenseMatrix<float> target{0,0};

    void compile(Layer* last, size_t batch_size){
        loss.malloc<data::BOTH>();
        target = data::DenseMatrix<float>{last->size, batch_size};
        target.malloc<data::BOTH>();
    }

    virtual void apply(Tape& output) = 0;
};

}    // namespace nn