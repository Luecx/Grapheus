#pragma once

#include "bitboard.h"
#include "square.h"

namespace chess {

struct PositionMetaInformation {

    uint8_t m_move_count;
    uint8_t m_fifty_move_rule;
    uint8_t m_castling_and_active_player {};
    Square  m_en_passant_square {N_SQUARES};

    Color   stm() const {
        return has(m_castling_and_active_player, 7);
    }

    void set_stm(Color color) {
        if (color) {
            m_castling_and_active_player |= (1 << 7);
        } else {
            m_castling_and_active_player &= ~(1 << 7);
        }
    }

    Square ep_square() const {
        return m_en_passant_square;
    }

    void set_ep_square(Square ep_square) {
        m_en_passant_square = ep_square;
    }

    bool can_castle(Color player, Side side) const {
        return m_castling_and_active_player & (1 << (player * 2 + side));
    }

    void set_castle(Color player, Side side, bool value) {
        if (value)
            m_castling_and_active_player |= (1 << (player * 2 + side));
        else
            m_castling_and_active_player &= ~(1 << (player * 2 + side));
    }

    uint8_t fifty_mr() const {
        return m_fifty_move_rule;
    }
    void set_fifty_mr(uint8_t p_fifty_move_rule) {
        m_fifty_move_rule = p_fifty_move_rule;
    }

    uint8_t move_count() const {
        return m_move_count;
    }
    void set_move_count(uint8_t p_move_count) {
        m_move_count = p_move_count;
    }
};

}    // namespace chess
