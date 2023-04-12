#pragma once

#include "../../data/matrix_dense.h"

namespace operations {

// clang-format off
__global__ void mpe_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
          float* __restrict__ loss,
    float power,
    float scale,
    size_t m,
    size_t n,
    size_t ldo,
    size_t ldt);

template<data::Device DEV>
inline void mpe(const data::DenseMatrix<float> &out,
                      data::DenseMatrix<float> &out_grd,
                const data::DenseMatrix<float> &target,
                const data::SArray     <float> &loss,
                float power,
                float scale){
    // clang-format on
    ASSERT(out.m == out_grd.m && out.m == target.m);
    ASSERT(out.n == out_grd.n && out.n == target.n);
    ASSERT(out.address<DEV>());
    ASSERT(out_grd.address<DEV>());
    ASSERT(target.address<DEV>());
    ASSERT(loss.address<DEV>());
    ASSERT(power > 0);
    ASSERT(scale > 0);

    if (data::is_gpu(DEV)) {
        constexpr size_t block_size_x = 32;
        constexpr size_t block_size_y = 32;
        dim3 block(block_size_x, block_size_y);
        dim3 grid (std::ceil((float)out.n / block_size_x),
                   std::ceil((float)out.m / block_size_x));
        mpe_kernel<<<grid, block>>>(
            out.first<DEV>(),
            out_grd.first<DEV>(),
            target.first<DEV>(),
            loss.address<DEV>(),
            power,
            scale,
            out.m,
            out.n,
            out.ld,
            target.ld);
    } else {
        ERROR(false);
    }
}

}    // namespace operations