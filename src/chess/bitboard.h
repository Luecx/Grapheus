#pragma once
#include "defs.h"

#include <iostream>
#ifndef __ARM__
#ifndef NO_IMMINTRIN
#include <immintrin.h>
#endif
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#pragma intrinsic(__popcnt64) // For MSVC, this ensures the intrinsic is available.
#endif

namespace chess{

/**
 * toggles the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void toggle(BB& number, Square index) {
    number ^= (1ULL << index);
}

/**
 * set the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void set(BB& number, Square index) {
    number |= (1ULL << index);
}

/**
 * unset the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline void unset(BB& number, Square index) {
    number &= ~(1ULL << index);
}

/**
 * get the bit
 * @param number    number to manipulate
 * @param index     index of bit starting at the LST
 * @return          the manipulated number
 */
inline bool has(BB number, Square index) {
    return ((number >> index) & 1ULL) == 1;
}

/**
 * returns the index of the LSB
 * @param bb
 * @return
 */
inline Square lsb(BB bb) {
//    UCI_ASSERT(bb != 0);
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return _tzcnt_u64(bb);
#else
    return __builtin_ctzll(bb);
#endif
}


/**
 * returns the amount of set bits in the given bitboard.
 * @param bb
 * @return
 */
inline int popcount(BB bb) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    return __popcnt64(bb);
#else
    return __builtin_popcountll(bb);
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
 * returns the index of the nth set bit, starting at the lsb
 * @param bb
 * @return
 */
inline Square nlsb(BB bb, Square n) {

//#ifdef __ARM__
    // Scatter the bit '1ULL << n' across the bits of 'bb'
    BB result = 0;
    for (BB mask = 1; bb != 0; mask <<= 1) {
        if (bb & 1) {
            if (n == 0) {
                result |= mask;
                break;
            }
            --n;
        }
        bb >>= 1;
    }
    return lsb(result);
//#else
//    return lsb(_pdep_u64(1ULL << n, bb));
//#endif
}



/**
 * resets the lsb in the given number and returns the result.
 * @param number
 * @return
 */
inline BB lsb_reset(BB number) {
    return number & (number - 1);
}

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

}    // namespace chess
