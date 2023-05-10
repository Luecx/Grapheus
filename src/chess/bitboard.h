#pragma once
#include "defs.h"

#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace chess{

/**
 * returns the amount of set bits in the given bitboard.
 * @param bb
 * @return
 */
inline int popcount(BB bb) { 
#if defined(_MSC_VER) 

    return (uint8_t)__popcnt64(bb);

#else

    return __builtin_popcountll(bb);

#endif
}
    
/**
 * toggles the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void toggle(BB& number, Square index) { number ^= (1ULL << index); }

/**
 * set the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void set(BB& number, Square index) { number |= (1ULL << index); }

/**
 * unset the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void unset(BB& number, Square index) { number &= ~(1ULL << index); }

/**
 * get the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline bool has(BB number, Square index) { return ((number >> index) & 1ULL) == 1; }

#if defined(__GNUC__) // GCC, Clang, ICC

inline Square lsb(BB b)
{
    if (!b)
        return 0;
    return Square(__builtin_ctzll(b));
}

inline Square msb(BB b)
{
    if (!b)
        return 0;
    return Square(63 ^ __builtin_clzll(b));
}

#elif defined(_MSC_VER) // MSVC

#ifdef _WIN64 // MSVC, WIN64
inline Square lsb(BB b)
{
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return (Square)idx;
}

inline Square msb(BB b)
{
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return (Square)idx;
}

#endif

#else

#error "Compiler not supported."

#endif

/**
 * returns the index of the nth set bit, starting at the lsb
 * @param bb
 * @return
 */inline Square nlsb(BB bb, Square n) {
// https://stackoverflow.com/questions/7669057/find-nth-set-bit-in-an-int
#ifdef __BMI2__
    return lsb(__builtin_ia32_pdep_di(1ULL << n, bb));
#else


    n += 1;
    BB shifted = 0;    // running total
    BB nBits;          // value for this iteration

    // handle no solution
    if (n > popcount(bb))
        return 64;

    while (n > 7) {
        // for large n shift out lower n-1 bits from v.
        nBits = n - 1;
        n -= popcount(bb & ((1 << nBits) - 1));
        bb >>= nBits;
        shifted += nBits;
    }

    BB next;
    // n is now small, clear out n-1 bits and return the next bit
    // v&(v-1): a well known software trick to remove the lowest set bit.
    while (next = bb & (bb - 1), --n) {
        bb = next;
    }
#ifdef __GNUC__
    return __builtin_ctzll((bb ^ next) << shifted);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, (bb ^ next) << shifted);
    return (Square)idx;
#endif

#endif
}


/**
 * counts the ones inside the bitboard before the given index
 */
inline int popcount(BB bb, int pos) {
    BB mask = ((BB) 1 << pos) - 1;
    return popcount(bb & mask);
}

/**
 * resets the lsb in the given number and returns the result.
 * @param number
 * @return
 */
inline BB lsb_reset(BB number) { return number & (number - 1); }

/**
 * find fully set groups of 4
 */
inline BB highlight_groups_of_4(BB bb) {
    bb &= (bb >> 1);
    bb &= (bb >> 2);
    return bb;
}

/**
 * stream bits of 4
 */
template<uint8_t values>
constexpr inline BB repeat_groups_of_4() {
    BB bb {};
    bb |= (BB) values & 0xF;
    bb |= (bb << 32);
    bb |= (bb << 16);
    bb |= (bb << 8);
    bb |= (bb << 4);
    return bb;
}

/**
 * prints the given bitboard as a bitmap to the standard output stream
 * @param bb
 */
inline void print_bb(BB bb) {
    for (int i = 7; i >= 0; i--) {
        for (int n = 0; n < 8; n++) {
            if ((bb >> (i * 8 + n)) & (BB) 1) {
                std::cout << "1";
            } else {
                std::cout << "0";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

/**
 * prints the given bits starting at the msb and ending at the lsb
 * @param bb
 */
inline void print_bits(BB bb) {
    for (int i = 63; i >= 0; i--) {
        if (has(bb, i))
            std::cout << "1";
        else
            std::cout << "0";

        if (i % 8 == 0)
            std::cout << " ";
    }
    std::cout << "\n";
}

/**
 *
 */
template<unsigned N, typename T = BB>
inline T mask() {
    return (T) (((T) 1 << N) - 1);
}

}

