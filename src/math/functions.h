//
// Created by Luecx on 25.03.2023.
//

#pragma once
#include "../data/matrix_dense.h"

#include <cmath>

namespace math {

template<typename TYPE, typename FUNC>
inline data::DenseMatrix<TYPE> apply(size_t p_m, size_t p_n, FUNC f) {
    data::DenseMatrix<TYPE> res {p_m, p_n};
    res.template malloc<data::CPU>();

    for (size_t m = 0; m < p_m; m++) {
        for (size_t n = 0; n < p_n; n++) {
            res(m, n) = f(m, n);
        }
    }

    return res;
}

#define SIMPLE_FUNCTION(name)                                                                        \
    template<typename TYPE>                                                                          \
    inline data::DenseMatrix<TYPE> name(const data::DenseMatrix<TYPE>& base) {                       \
        return apply<TYPE>(base.m, base.n, [&base](size_t m, size_t n) {                             \
            return std::name(base(m, n));                                                            \
        });                                                                                          \
    }

#define FUNCTION_ONE_V_ARGUMENT(name)                                                                \
    template<typename TYPE>                                                                          \
    inline data::DenseMatrix<TYPE> name(const data::DenseMatrix<TYPE>& base, TYPE val) {             \
        return apply<TYPE>(base.m, base.n, [&base, &val](size_t m, size_t n) {                       \
            return std::name(base(m, n), val);                                                       \
        });                                                                                          \
    }

#define FUNCTION_ONE_M_ARGUMENT(name)                                                                \
    template<typename TYPE>                                                                          \
    inline data::DenseMatrix<TYPE> name(const data::DenseMatrix<TYPE>& base,                         \
                                        const data::DenseMatrix<TYPE>& val) {                        \
        return apply<TYPE>(base.m, base.n, [&base, &val](size_t m, size_t n) {                       \
            return std::name(base(m, n), val(m, n));                                                 \
        });                                                                                          \
    }

template<typename TYPE>
inline data::DenseMatrix<TYPE> pow(const data::DenseMatrix<TYPE>& base,
                                   const data::DenseMatrix<TYPE>& exp) {
    return apply(base.m, base.n, [&base, &exp](size_t m, size_t n) {
        return std::pow(base(m, n), exp(m, n));
    });
}

template<typename TYPE>
inline data::DenseMatrix<TYPE> pow(const data::DenseMatrix<TYPE>& base, TYPE exp) {
    return apply<TYPE>(base.m, base.n, [&base, &exp](size_t m, size_t n) {
        return std::pow(base(m, n), exp);
    });
}

SIMPLE_FUNCTION(sin);
SIMPLE_FUNCTION(cos);
SIMPLE_FUNCTION(tan);
SIMPLE_FUNCTION(asin);
SIMPLE_FUNCTION(acos);
SIMPLE_FUNCTION(atan)
SIMPLE_FUNCTION(atan2);

SIMPLE_FUNCTION(sinh);
SIMPLE_FUNCTION(cosh);
SIMPLE_FUNCTION(tanh);
SIMPLE_FUNCTION(asinh);
SIMPLE_FUNCTION(acosh);
SIMPLE_FUNCTION(atanh);

SIMPLE_FUNCTION(erf);
SIMPLE_FUNCTION(erfc);

SIMPLE_FUNCTION(tgamma);
SIMPLE_FUNCTION(lgamma);

SIMPLE_FUNCTION(ceil);
SIMPLE_FUNCTION(floor);
SIMPLE_FUNCTION(trunc)
SIMPLE_FUNCTION(round);

SIMPLE_FUNCTION(exp);
SIMPLE_FUNCTION(exp2);
SIMPLE_FUNCTION(log)
SIMPLE_FUNCTION(log10);
SIMPLE_FUNCTION(log2);
SIMPLE_FUNCTION(sqrt);
SIMPLE_FUNCTION(cbrt);

SIMPLE_FUNCTION(abs);
SIMPLE_FUNCTION(fmod);
SIMPLE_FUNCTION(remainder);

FUNCTION_ONE_V_ARGUMENT(max);
FUNCTION_ONE_M_ARGUMENT(max);
FUNCTION_ONE_V_ARGUMENT(fmax);
FUNCTION_ONE_M_ARGUMENT(fmax);

FUNCTION_ONE_V_ARGUMENT(min);
FUNCTION_ONE_M_ARGUMENT(min);
FUNCTION_ONE_V_ARGUMENT(fmin);
FUNCTION_ONE_M_ARGUMENT(fmin);

#undef TRIGONOMETRIC_FUNCTION

}    // namespace math
