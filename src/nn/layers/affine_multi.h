#pragma once

#include "affine.h"

namespace nn {

struct AffineMulti : public nn::Affine {

    AffineMulti(Layer* prev, size_t size, size_t counts)
        : Affine(prev, size * counts) {}
};
}    // namespace nn
