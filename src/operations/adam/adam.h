#pragma once

namespace operations {

__global__ void adam_kernel(float* __restrict__ values,
                            float* __restrict__ gradients,
                            float* __restrict__ first_moment,
                            float* __restrict__ second_moment,
                            size_t m,
                            size_t n,
                            size_t ldv,
                            size_t ldm,
                            float  alpha,
                            float  beta1,
                            float  beta2,
                            float  eps,
                            float  m_min,
                            float  m_max,
                            float  lasso,
                            float  ridge) {
    // clang-format on

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= n || idy >= m)
        return;

    size_t idv = MATRIX_INDEX(ldv, idy, idx);
    size_t idm = MATRIX_INDEX(ldm, idy, idx);

    // adjust gradient using lasso
    gradients[idv] += values[idv] > 0 ? lasso : values[idv] < 0 ? -lasso : 0;
    // adjust gradient using ridge
    gradients[idv] += 2 * ridge * values[idv];

    // standard adam
    first_moment[idm]  = beta1 * first_moment [idm] + (1 - beta1) * gradients[idv];
    second_moment[idm] = beta2 * second_moment[idm] + (1 - beta2) * gradients[idv] * gradients[idv];

    float delta        = alpha * first_moment[idm] / (sqrtf(second_moment[idm]) + eps);
    values[idv]        = max(m_min, min(m_max, values[idv] - delta));
    gradients[idv]     = 0;
}

// clang-format off
template<data::Device DEV>
void adam(      data::DenseMatrix<float>& values,
          const data::DenseMatrix<float>& gradients,
                data::DenseMatrix<float>& first_moment,
                data::DenseMatrix<float>& second_moment,
          float                           lr,
          float                           beta1,
          float                           beta2,
          float                           eps,
          // not directly related to adam but any optimizer should set use this too ^^
          float min,
          float max,
          float lasso,
          float ridge) {
    // clang-format on

    ASSERT(first_moment.ld == second_moment.ld);
    ASSERT(values.ld == gradients.ld);
    ASSERT(values.m == gradients.m && gradients.m == first_moment.m
           && first_moment.m == second_moment.m);
    ASSERT(values.n == gradients.n && gradients.n == first_moment.n
           && first_moment.n == second_moment.n);


    ASSERT(values.first<DEV>());
    ASSERT(gradients.first<DEV>());
    ASSERT(first_moment.first<DEV>());
    ASSERT(second_moment.first<DEV>());

    if (data::is_gpu(DEV)) {
        int block_size_x;
        int block_size_y;
        if (values.m > 128) {
            block_size_x = 1;
            block_size_y = 32;
        } else if (values.m > 8) {
            block_size_x = 32;
            block_size_y = 8;
        } else {
            block_size_x = 512;
            block_size_y = 1;
        };

        dim3 block(block_size_x, block_size_y);
        dim3 grid(std::ceil((float) values.n / block_size_x),
                  std::ceil((float) values.m / block_size_y));
        adam_kernel<<<grid, block>>>(values.first<DEV>(),
                                      gradients.first<DEV>(),
                                      first_moment.first<DEV>(),
                                      second_moment.first<DEV>(),
                                      values.m,
                                      values.n,
                                      values.ld,
                                      first_moment.ld,
                                      lr,
                                      beta1,
                                      beta2,
                                      eps,
                                      min,
                                      max,
                                      lasso,
                                      ridge);
    } else {
        ERROR(false);
    }
}

}    // namespace operations