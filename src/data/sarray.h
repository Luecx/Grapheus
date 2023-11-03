//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include "array.h"
#include "device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <new>
#include <ostream>

namespace data {

template<typename TYPE = float>
class SArray : public Array<TYPE> {
    using CPUArrayTYPE = CPUArray<TYPE>;
    using GPUArrayTYPE = GPUArray<TYPE>;

    using CPtr         = std::shared_ptr<CPUArrayTYPE>;
    using GPtr         = std::shared_ptr<GPUArrayTYPE>;

    protected:
    CPtr cpu_values = nullptr;
    GPtr gpu_values = nullptr;

    // construction / deconstruction
    public:
    explicit SArray(size_t p_size);
    SArray(const SArray<TYPE>& other);
    SArray(SArray<TYPE>&& other) noexcept;
    SArray<TYPE>& operator=(const SArray<TYPE>& other);
    SArray<TYPE>& operator=(SArray<TYPE>&& other) noexcept;
    virtual ~SArray();

    // memory managing function
    template<Device DEV>
    inline void free();
    template<Device DEV>
    inline void malloc();
    template<Device DEV>
    inline bool is_allocated() const;
    template<Device DEV>
    inline TYPE* address() const;
    template<Device DEV>
    inline void sync();

    // copy from another array
    template<Device DEV = CPU>
    inline void copy_from(const SArray& other);

    // sync using operators
    inline SArray& operator>>(Device d);
    inline SArray& operator<<(Device d);

    // gets values from the cpu memory
    inline TYPE  get(size_t height) const;
    inline TYPE& get(size_t height);
    inline TYPE  operator()(size_t height) const;
    inline TYPE& operator()(size_t height);
    inline TYPE  operator[](size_t height) const;
    inline TYPE& operator[](size_t height);

    // basic element wise operators
    // only performed on the cpu and not on the gpu
    inline SArray<TYPE>  operator+(const SArray<TYPE>& other);
    inline SArray<TYPE>& operator+=(const SArray<TYPE>& other);
    inline SArray<TYPE>  operator-(const SArray<TYPE>& other);
    inline SArray<TYPE>& operator-=(const SArray<TYPE>& other);
    inline SArray<TYPE>  operator*(const SArray<TYPE>& other);
    inline SArray<TYPE>& operator*=(const SArray<TYPE>& other);
    inline SArray<TYPE>  operator/(const SArray<TYPE>& other);
    inline SArray<TYPE>& operator/=(const SArray<TYPE>& other);
};

template<typename TYPE>
SArray<TYPE>::SArray(size_t p_size)
    : Array<TYPE>(p_size) {}

template<typename TYPE>
SArray<TYPE>::SArray(const SArray<TYPE>& other)
    : Array<TYPE>(other.m_size) {
    if (other.is_allocated<CPU>()) {
        malloc<CPU>();
        this->template copy_from<CPU>(other);
    }
    if (other.is_allocated<GPU>()) {
        malloc<GPU>();
        this->template copy_from<GPU>(other);
    }
}

template<typename TYPE>
SArray<TYPE>::SArray(SArray<TYPE>&& other) noexcept
    : Array<TYPE>(other.m_size) {
    this->cpu_values = other.cpu_values;
    this->gpu_values = other.gpu_values;
}

template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator=(const SArray<TYPE>& other) {
    free<BOTH>();
    this->m_size = other.m_size;
    if (other.is_allocated<CPU>()) {
        malloc<CPU>();
        this->template copy_from<CPU>(other);
    }
    if (other.is_allocated<GPU>()) {
        malloc<GPU>();
        this->template copy_from<GPU>(other);
    }
    return (*this);
}

template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator=(SArray<TYPE>&& other) noexcept {
    free<BOTH>();
    this->m_size     = other.m_size;
    this->cpu_values = other.cpu_values;
    this->gpu_values = other.gpu_values;
    return (*this);
}

template<typename TYPE>
SArray<TYPE>::~SArray() {
    free<BOTH>();
}

template<typename TYPE>
template<Device DEV>
void SArray<TYPE>::free() {
    static_assert(DEV != NONE);
    if constexpr (is_gpu(DEV)) {
        gpu_values = nullptr;
    }
    if constexpr (is_cpu(DEV)) {
        cpu_values = nullptr;
    }
}

template<typename TYPE>
template<Device DEV>
void SArray<TYPE>::malloc() {
    static_assert(DEV != NONE);
    if constexpr (is_gpu(DEV)) {
        if (is_allocated<DEV>()) {
            free<DEV>();
        }
        gpu_values = GPtr(new GPUArrayTYPE(this->m_size));
    }
    if constexpr (is_cpu(DEV)) {
        if (is_allocated<DEV>()) {
            free<DEV>();
        }
        cpu_values = CPtr(new CPUArrayTYPE(this->m_size));
    }
}

template<typename TYPE>
template<Device DEV>
bool SArray<TYPE>::is_allocated() const {
    static_assert(DEV != NONE);
    if constexpr (is_gpu(DEV)) {
        if (gpu_values == nullptr)
            return false;
    }
    if constexpr (is_cpu(DEV)) {
        if (cpu_values == nullptr)
            return false;
    }
    return true;
}

template<typename TYPE>
template<Device DEV>
TYPE* SArray<TYPE>::address() const {
    static_assert(DEV != BOTH);
    static_assert(DEV != NONE);
    if (is_allocated<DEV>()) {
        if constexpr (DEV == CPU) {
            return cpu_values->m_data;
        } else {
            return gpu_values->m_data;
        }
    }
    return nullptr;
}

template<typename TYPE>
template<Device DEV>
void SArray<TYPE>::sync() {
    static_assert(DEV != BOTH);
    static_assert(DEV != NONE);

    if (!is_allocated<CPU>()) {
        malloc<CPU>();
    }
    if (!is_allocated<GPU>()) {
        malloc<GPU>();
    }

    if constexpr (DEV == CPU) {
        gpu_values->download(*cpu_values.get());
    } else {
        gpu_values->upload(*cpu_values.get());
    }
}

template<typename TYPE>
template<Device DEV>
void SArray<TYPE>::copy_from(const SArray& other) {
    ASSERT(other.is_allocated<DEV>());
    // make sure that memory is allocated
    if (!is_allocated<DEV>()) {
        malloc<DEV>();
    }
    // do the copy
    if constexpr (is_cpu(DEV)) {
        cpu_values->copy_from(*other.cpu_values.get());
    }
    if constexpr (is_gpu(DEV)) {
        gpu_values->copy_from(*other.gpu_values.get());
    }
}

template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator>>(Device d) {
    if (is_cpu(d)) {
        sync<CPU>();
    } else {
        sync<GPU>();
    }
    return *this;
}
template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator<<(Device d) {
    if (is_cpu(d)) {
        sync<GPU>();
    } else {
        sync<CPU>();
    }
    return *this;
}
template<typename TYPE>
TYPE SArray<TYPE>::get(size_t height) const {
    ASSERT(is_allocated<CPU>());
    ASSERT(height < this->size());
    return cpu_values->m_data[height];
}
template<typename TYPE>
TYPE& SArray<TYPE>::get(size_t height) {
    ASSERT(is_allocated<CPU>());
    ASSERT(height < this->size());
    return cpu_values->m_data[height];
}
template<typename TYPE>
TYPE SArray<TYPE>::operator()(size_t height) const {
    ASSERT(is_allocated<CPU>());
    return get(height);
}
template<typename TYPE>
TYPE& SArray<TYPE>::operator()(size_t height) {
    ASSERT(is_allocated<CPU>());
    return get(height);
}
template<typename TYPE>
TYPE SArray<TYPE>::operator[](size_t height) const {
    ASSERT(is_allocated<CPU>());
    return get(height);
}
template<typename TYPE>
TYPE& SArray<TYPE>::operator[](size_t height) {
    ASSERT(is_allocated<CPU>());
    return get(height);
}
template<typename TYPE>
SArray<TYPE> SArray<TYPE>::operator+(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    SArray<TYPE> res {*this};
    res += other;
    return res;
}
template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator+=(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    for (size_t i = 0; i < this->size(); i++) {
        this->get(i) += other.get(i);
    }
    return *this;
}
template<typename TYPE>
SArray<TYPE> SArray<TYPE>::operator-(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    SArray<TYPE> res {*this};
    res -= other;
    return res;
}
template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator-=(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    for (size_t i = 0; i < this->size(); i--) {
        this->get(i) -= other.get(i);
    }
    return *this;
}
template<typename TYPE>
SArray<TYPE> SArray<TYPE>::operator*(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    SArray<TYPE> res {*this};
    res *= other;
    return res;
}
template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator*=(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    for (size_t i = 0; i < this->size(); i--) {
        this->get(i) *= other.get(i);
    }
    return *this;
}
template<typename TYPE>
SArray<TYPE> SArray<TYPE>::operator/(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    SArray<TYPE> res {*this};
    res /= other;
    return res;
}
template<typename TYPE>
SArray<TYPE>& SArray<TYPE>::operator/=(const SArray<TYPE>& other) {
    ASSERT(other.size() == this->size());
    ASSERT(is_allocated<CPU>());
    ASSERT(other.is_allocated<CPU>());
    for (size_t i = 0; i < this->size(); i--) {
        this->get(i) /= other.get(i);
    }
    return *this;
}

}    // namespace data

