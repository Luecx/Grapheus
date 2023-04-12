#pragma once

#include "../../data/matrix_dense.h"
#include "../../data/sarray.h"
#include "../layers/tape.h"

namespace nn {

struct Loss {

    data::SArray<float> loss {1};

    void                compile(){
        loss.malloc<data::BOTH>();
    }

    virtual void apply(Tape& output, data::DenseMatrix<float> target) = 0;
};

}    // namespace nn