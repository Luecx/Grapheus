//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include "device.h"
#include "matrix.h"
#include "sarray.h"

#include <iostream>
#include <functional>

namespace data {
template<typename TYPE = float>
struct DenseMatrix : public SArray<TYPE>, Matrix {
    size_t ld;                 // leading dimension
    size_t offset;             // offset for first entry in case of sub-matrices
    bool submat;               // flag for detecting if this matrix is a submatrix
                               // relevant copy construction / move construction / assignment

    // construction
    DenseMatrix(const size_t& m, const size_t& n);
    DenseMatrix(const DenseMatrix<TYPE>& other,
                size_t                   m,
                size_t                   n,
                size_t                   offset_m = 0,
                size_t                   offset_n = 0);
    DenseMatrix(const DenseMatrix<TYPE>& other);
    DenseMatrix(DenseMatrix<TYPE>&& other);
    DenseMatrix<TYPE>& operator=(const DenseMatrix<TYPE>& other);
    DenseMatrix<TYPE>& operator=(DenseMatrix<TYPE>&& other);

    // pointer to the very first entry
    template<Device DEV>
    TYPE* first() const;

    // getters and setters using two indices
    TYPE& get(int p_m, int p_n);
    TYPE  get(int p_m, int p_n) const;
    TYPE  operator()(int p_m, int p_n) const;
    TYPE& operator()(int p_m, int p_n);

    // getters and setters using the SArray indexing
    using SArray<TYPE>::get;
    using SArray<TYPE>::operator();
    using SArray<TYPE>::operator[];

    // printing
    template<typename TYPE_>
    friend std::ostream& operator<<(std::ostream& os, const DenseMatrix<TYPE>& data);

    // basic element wise operators
    // only performed on the cpu and not on the gpu
    inline DenseMatrix<TYPE>  operator+(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>& operator+=(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>  operator-(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>& operator-=(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>  operator*(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>& operator*=(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>  operator/(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>& operator/=(const DenseMatrix<TYPE>& other);
    inline DenseMatrix<TYPE>  operator-() const;
    inline DenseMatrix<TYPE>& operator*=(TYPE val);
    inline DenseMatrix<TYPE>  operator*(TYPE val);
    inline DenseMatrix<TYPE>& operator+=(TYPE val);
    inline DenseMatrix<TYPE>  operator+(TYPE val);
    inline DenseMatrix<TYPE>& operator-=(TYPE val);
    inline DenseMatrix<TYPE>  operator-(TYPE val);
    inline DenseMatrix<TYPE>& operator/=(TYPE val);
    inline DenseMatrix<TYPE>  operator/(TYPE val);
    inline DenseMatrix<TYPE>& operator=(TYPE value);

    // for each function which allows easy iteration over the content
    inline void for_each(std::function<void(size_t, size_t, TYPE&)> func);
};


template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(const size_t& m, const size_t& n)
    : SArray<TYPE>(m * n)
    , Matrix(m, n)
    , ld {m}
    , offset {0}
    , submat {false}{}

template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(const DenseMatrix& other,
                               size_t             m,
                               size_t             n,
                               size_t             offset_m,
                               size_t             offset_n)
    : SArray<TYPE>(other.m_size)
    , Matrix(m, n)
    , ld {other.ld}
    , submat {true}
    , offset {other.ld * offset_n + offset_m + other.offset} {
    // manually invoke the copy here
    this->cpu_values = other.cpu_values;
    this->gpu_values = other.gpu_values;
}

template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(const DenseMatrix<TYPE>& other)
    : SArray<TYPE>(other.m_size)
    , Matrix(other.m, other.n)
    , ld {other.ld}
    , submat {other.submat}
    , offset {other.offset}{
    if (other.submat) {
        this->m_size     = other.m_size;
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
    } else {
        SArray<TYPE>::operator=(other);
    }
}

template<typename TYPE>
DenseMatrix<TYPE>::DenseMatrix(DenseMatrix<TYPE>&& other)
    : SArray<TYPE>(other)
    , Matrix(other.m, other.n)
    , ld {other.ld}
    , submat {other.submat}
    , offset {other.offset} {

}

template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator=(const DenseMatrix<TYPE>& other) {
    ld     = other.ld;
    submat = other.submat;
    m      = other.m;
    n      = other.n;
    offset = other.offset;

    if (other.submat) {
        this->m_size     = other.m_size;
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
    } else {
        SArray<TYPE>::operator=(other);
    }
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator=(DenseMatrix<TYPE>&& other) {
    ld     = other.ld;
    submat = other.submat;
    m      = other.m;
    n      = other.n;
    offset = other.offset;

    if (other.submat) {
        this->m_size     = other.m_size;
        this->cpu_values = other.cpu_values;
        this->gpu_values = other.gpu_values;
    } else {
        SArray<TYPE>::operator=(other);
    }
    return *this;
}


template<typename TYPE>
template<Device DEV>
TYPE* DenseMatrix<TYPE>::first() const {
    static_assert(DEV != BOTH);
    static_assert(DEV != NONE);
    if (this->template is_allocated<DEV>()) {
        if constexpr (DEV == CPU) {
            return this->cpu_values->m_data + offset;
        } else {
            return this->gpu_values->m_data + offset;
        }
    }
    return nullptr;
}

template<typename TYPE>
TYPE& DenseMatrix<TYPE>::get(int p_m, int p_n) {
    return SArray<TYPE>::get(offset + MATRIX_INDEX(ld, p_m, p_n));
}
template<typename TYPE>
TYPE DenseMatrix<TYPE>::get(int p_m, int p_n) const {
    return SArray<TYPE>::get(offset + MATRIX_INDEX(ld, p_m, p_n));
}
template<typename TYPE>
TYPE DenseMatrix<TYPE>::operator()(int p_m, int p_n) const {
    return get(p_m, p_n);
}
template<typename TYPE>
TYPE& DenseMatrix<TYPE>::operator()(int p_m, int p_n) {
    return get(p_m, p_n);
}

template<typename TYPE_>
std::ostream& operator<<(std::ostream& os, const DenseMatrix<TYPE_>& data) {
//    os << "size       : " << data.size() << "\n"
//       << "CPU address: " << data.template address<CPU>() << " + " << data.offset << "\n"
//       << "GPU address: " << data.template address<GPU>() << " + " << data.offset << "\n";

    if (data.n != 1) {
        os << std::fixed << std::setprecision(10);
        for (int p_i = 0; p_i < data.m; p_i++) {
            for (int p_n = 0; p_n < data.n; p_n++) {
                os << std::setw(20) << (double) data(p_i, p_n);
            }
            os << "\n";
        }
    } else {
        os << "(transposed) ";
        for (int n = 0; n < data.m; n++) {
            os << std::setw(11) << (double) data(n, 0);
        }
        os << "\n";
    }
    return os;
}

template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator+(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(this->is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    DenseMatrix<TYPE> res {*this};
    res += other;
    return res;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator+=(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    for (size_t m = 0; m < this->m; m++) {
        for (size_t n = 0; n < this->n; n++) {
            this->get(m, n) += other.get(m, n);
        }
    }
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator-(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    DenseMatrix<TYPE> res {*this};
    res -= other;
    return res;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator-=(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    for (size_t m = 0; m < this->m; m++) {
        for (size_t n = 0; n < this->n; n++) {
            this->get(m, n) -= other.get(m, n);
        }
    }
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator*(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    DenseMatrix<TYPE> res {*this};
    res *= other;
    return res;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator*=(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    for (size_t m = 0; m < this->m; m++) {
        for (size_t n = 0; n < this->n; n++) {
            this->get(m, n) *= other.get(m, n);
        }
    }
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator/(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    //    ASSERT(is_allocated<CPU>());
    //    ASSERT(other.is_allocated<CPU>());
    DenseMatrix<TYPE> res {*this};
    res /= other;
    return res;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator/=(const DenseMatrix<TYPE>& other) {
    ASSERT(other.m == this->m);
    ASSERT(other.n == this->n);
    ASSERT(this->template is_allocated<CPU>());
    ASSERT(other.template is_allocated<CPU>());
    for (size_t m = 0; m < this->m; m++) {
        for (size_t n = 0; n < this->n; n++) {
            this->get(m, n) /= other.get(m, n);
        }
    }
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator-() const {
    ASSERT(this->template is_allocated<CPU>());
    DenseMatrix<TYPE> res {*this};
    for (size_t m = 0; m < this->m; m++) {
        for (size_t n = 0; n < this->n; n++) {
            this->get(m, n) = -get(m, n);
        }
    }
    return res;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator*=(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    for_each([&val](size_t m, size_t n, float& v) {v *= val;});
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator*(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    return DenseMatrix<TYPE>(*this) *= val;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator+=(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    for_each([&val](size_t m, size_t n, float& v) {v += val;});
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator+(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    return DenseMatrix<TYPE>(*this) += val;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator-=(TYPE val) {
    for_each([&val](size_t m, size_t n, float& v) {v -= val;});
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator-(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    return DenseMatrix<TYPE>(*this) -= val;
}
template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator/=(TYPE val) {
    for_each([&val](size_t m, size_t n, float& v) {v /= val;});
    return *this;
}
template<typename TYPE>
DenseMatrix<TYPE> DenseMatrix<TYPE>::operator/(TYPE val) {
    ASSERT(this->template is_allocated<CPU>());
    return DenseMatrix<TYPE>(*this) /= val;
}

template<typename TYPE>
DenseMatrix<TYPE>& DenseMatrix<TYPE>::operator=(TYPE value){
    for_each([&value](size_t m, size_t n, float& v) {v = value;});
    return *this;
}

template<typename TYPE>
void DenseMatrix<TYPE>::for_each(std::function<void(size_t, size_t, TYPE&)> func) {
    ASSERT(this->template is_allocated<CPU>());
    for (size_t i = 0; i < this->m; ++i) {
        for (size_t j = 0; j < this->n; ++j) {
            func(i, j, this->get(i, j));
        }
    }
}

}    // namespace data
