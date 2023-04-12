#pragma once

#include "mpe.h"
namespace nn {

struct MSE : public MPE {

    MSE(bool average_gradients = true)
        : MPE(2, average_gradients) {}
};
}    // namespace nn
