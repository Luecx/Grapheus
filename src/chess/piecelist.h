#pragma once

#include <iomanip>

namespace chess {

#define MAX_PIECES_PER_BOARD (32)
#define PIECES_PER_BUCKET    (64 / 4)
#define MAX_BUCKETS          (MAX_PIECES_PER_BOARD / PIECES_PER_BUCKET)

/**
 * container storing the pieces on the board
 */
struct PieceList {

    BB    m_piece_buckets[MAX_BUCKETS] {};

    Piece get_piece(const int index) const {
        // compute the bucket and the offset
        const int bucket = index / PIECES_PER_BUCKET;
        const int offset = index % PIECES_PER_BUCKET;

        // shift the content of the bucket and isolate the last 4 bits
        return (m_piece_buckets[bucket] >> (4 * offset)) & mask<4>();
    }

    void set_piece(const int index, const Piece piece) {
        // compute the bucket and the offset
        const int bucket = index / PIECES_PER_BUCKET;
        const int offset = index % PIECES_PER_BUCKET;

        // compute the shift
        const int shift = 4 * offset;

        // disable all in the correct place
        m_piece_buckets[bucket] &= ~(mask<4>() << shift);

        // set all in the correct place
        m_piece_buckets[bucket] |= ((BB) piece << shift);
    }

    friend std::ostream& operator<<(std::ostream& os, const PieceList& list) {
        for (int i = 0; i < MAX_BUCKETS; i++) {
            std::cout << "bucket " << std::setw(2) << i << " | ";
            for (int b = 0; b < PIECES_PER_BUCKET; b++) {
                std::cout << has(list.m_piece_buckets[i], b * 4 + 3);
                std::cout << has(list.m_piece_buckets[i], b * 4 + 2);
                std::cout << has(list.m_piece_buckets[i], b * 4 + 1);
                std::cout << has(list.m_piece_buckets[i], b * 4 + 0);
                std::cout << " ";
            }
            std::cout << "\n";
        }
        return os;
    }

    template<Piece piece>
    int bitscan_piece() const {
        constexpr BB piecePattern = repeat_groups_of_4<piece>();
        constexpr BB lsbPattern   = repeat_groups_of_4<1>();

        constexpr BB invPiecePat  = ~piecePattern;

        // check the lower 16 entries first
        BB mask_lower = invPiecePat ^ m_piece_buckets[0];
        mask_lower    = highlight_groups_of_4(mask_lower);
        mask_lower &= lsbPattern;

        if (mask_lower) {
            return lsb(mask_lower) >> 2;    // alternative / 4
        }

        // if we havent found one in the first 16, check the next 16 pieces
        BB mask_upper = invPiecePat ^ m_piece_buckets[1];
        mask_upper    = highlight_groups_of_4(mask_upper);
        mask_upper &= lsbPattern;

        // make sure to add 16 (PIECES_PER_BUCKET)
        return (lsb(mask_upper) >> 2) + PIECES_PER_BUCKET;
    }
};
}    // namespace chess
