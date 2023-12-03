//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include "../misc/assert.h"

#include <cstring>
#include <iostream>

namespace data {
template<typename Type>
struct Array {
    size_t m_size = 0;
    Type*  m_data = nullptr;

    explicit Array(size_t p_size)
        : m_size(p_size) {}

    inline size_t size() const {
        return m_size;
    }
};

template<typename Type>
struct CPUArray : Array<Type> {
    explicit CPUArray(size_t p_size)
        : Array<Type>(p_size) {
#ifndef NDEBUG
        std::cout << "CPU allocating data" << std::endl;
#endif
        this->m_data = new Type[p_size] {};
    }
    virtual ~CPUArray() {
#ifndef NDEBUG
        std::cout << "CPU deleting data" << std::endl;
#endif
        delete[] this->m_data;
    }

    inline Type& operator()(size_t idx) {
        return this->m_data[idx];
    }
    inline Type operator()(size_t idx) const {
        return this->m_data[idx];
    }
    inline Type& operator[](size_t idx) {
        return this->m_data[idx];
    }
    inline Type operator[](size_t idx) const {
        return this->m_data[idx];
    }

    inline void copy_from(const CPUArray<Type>& other) {
#ifndef NDEBUG
        std::cout << "CPU copy" << std::endl;
#endif
        ASSERT(other.size() == this->size());
        memcpy(this->m_data, other.m_data, this->size() * sizeof(Type));
    }
    inline void clear() {
#ifndef NDEBUG
        std::cout << "CPU clear" << std::endl;
#endif
        memset(this->m_data, 0, sizeof(Type) * this->size());
    }
};

template<typename Type>
struct GPUArray : Array<Type> {
    explicit GPUArray(size_t p_size)
        : Array<Type>(p_size) {
#ifndef NDEBUG
        std::cout << "GPU allocating data" << std::endl;
#endif
        CUDA_ASSERT(cudaMalloc(&this->m_data, this->size() * sizeof(Type)));
    }
    virtual ~GPUArray() {
#ifndef NDEBUG
        std::cout << "GPU deleting data" << std::endl;
#endif
        CUDA_ASSERT(cudaFree(this->m_data));
    }

    void upload(CPUArray<Type>& cpu_array) {
#ifndef NDEBUG
        std::cout << "moving data from cpu to gpu" << std::endl;
#endif
        CUDA_ASSERT(cudaMemcpy(this->m_data,
                               cpu_array.m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyHostToDevice));
    }
    void download(CPUArray<Type>& cpu_array) {
#ifndef NDEBUG
        std::cout << "moving data from gpu to cpu" << std::endl;
#endif
        CUDA_ASSERT(cudaMemcpy(cpu_array.m_data,
                               this->m_data,
                               this->m_size * sizeof(Type),
                               cudaMemcpyDeviceToHost));
    }

    void copy_from(const GPUArray<Type>& other) {
#ifndef NDEBUG
        std::cout << "GPU copy" << std::endl;
#endif
        ASSERT(other.size() == this->size());
        cudaMemcpy(this->m_data, other.m_data, this->size() * sizeof(Type), cudaMemcpyDeviceToDevice);
    }
    void clear() {
#ifndef NDEBUG
        std::cout << "CPU clear" << std::endl;
#endif
        cudaMemset(this->m_data, 0, sizeof(Type) * this->size());
    }
};

}    // namespace data
