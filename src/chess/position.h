#pragma once

#include "bitboard.h"
#include "piece.h"
#include "piecelist.h"
#include "positionmeta.h"
#include "result.h"

#include <iomanip>
#include <ostream>

namespace chess{


struct Position : dataset::DataSetEntry{

    PieceList               m_pieces {};
    BB                      m_occupancy {};
    PositionMetaInformation m_meta {};
    Result                  m_result {};

    template<Color color>
    Square get_king_square() {
        if (color == WHITE) {
            return nlsb(m_occupancy, m_pieces.template bitscan_piece<WHITE_KING>());
        } else {
            return nlsb(m_occupancy, m_pieces.template bitscan_piece<BLACK_KING>());
        }
    }

    template<Piece piece>
    bool has_piece(){
        int idx = m_pieces.template bitscan_piece<piece>();
        return idx >= 0 && idx < 32;
    }

    int piece_count() const { return popcount(m_occupancy); }

    Square get_square(int piece_index) const { return nlsb(m_occupancy, piece_index); }

    Piece  get_piece(Square square) const {
        if (has(m_occupancy, square)) {
            return m_pieces.get_piece(popcount(m_occupancy, square));
        }
        return NO_PIECE;
    }
};

}
