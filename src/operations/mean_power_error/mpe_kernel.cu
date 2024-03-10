#include "mpe.h"

// clang-format off
__global__ void operations::mpe_kernel(
    const float* __restrict__ output,
          float* __restrict__ output_gradient,
    const float* __restrict__ target,
          float* __restrict__ loss,
    float power,
    float scale,
    size_t m,
    size_t n,
    size_t ldo,
    size_t ldt){
    // clang-format on

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= n || idy >= m)
        return;


    float difference     = output[MATRIX_INDEX(ldo, idy, idx)] - target[MATRIX_INDEX(ldt, idy, idx)];
    float abs_diff       = abs(difference);
    float sign           = difference > 0 ? 1 : -1;

    float derivative     = powf(abs_diff, power - 1) * sign * power;
    float loss_val       = powf(abs_diff, power);

    output_gradient[MATRIX_INDEX(ldo, idy, idx)] = derivative * scale;
    atomicAdd(loss, loss_val / (m * n));
}