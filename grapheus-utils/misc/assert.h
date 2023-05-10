//
// Created by Luecx on 18.03.2023.
//

#pragma once
#include <iostream>

#ifdef NDEBUG
#define ASSERT(expr)
#else
#define ASSERT(expr)                                                                                 \
    {                                                                                                \
        if (!static_cast<bool>(expr)) {                                                              \
            std::cout << "[ASSERT] in expression " << (#expr) << std::endl;                          \
            std::cout << "    file: " << __FILE__ << std::endl;                                      \
            std::cout << "    line: " << __LINE__ << std::endl;                                      \
            std::cout << "    func: " << __FUNCTION__ << std::endl;                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    }
#endif

#define ERROR(expr)                                                                                  \
    {                                                                                                \
        if (!static_cast<bool>(expr)) {                                                              \
            std::cout << "[ERROR] in expression " << (#expr) << std::endl;                           \
            std::cout << "    file: " << __FILE__ << std::endl;                                      \
            std::cout << "    line: " << __LINE__ << std::endl;                                      \
            std::cout << "    func: " << __FUNCTION__ << std::endl;                                  \
            std::exit(1);                                                                            \
        }                                                                                            \
    }