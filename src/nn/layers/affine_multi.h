#pragma once

#include "affine.h"

namespace nn {

struct AffineMulti : public nn::Affine {

    AffineMulti(Layer* prev, size_t size, size_t batches)
        : Affine(prev, size * batches) {

        for (size_t i = size; i < this->size; i++)
            for (size_t j = 0; j < prev->size; j++)
                weights.values(i, j) = weights.values(i % size, j);

        weights.values >> data::GPU;
        bias.values >> data::GPU;
    }
};

}    // namespace nn
