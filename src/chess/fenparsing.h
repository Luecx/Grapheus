#pragma once

#include "bitboard.h"
#include "piece.h"
#include "position.h"
#include "square.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

namespace chess {

struct FenCharacter {
    char   character {0};
    Square skip_squares {0};
    Piece  piece {0};
    Rank   rank {0};
    File   file {0};
};

static FenCharacter fen_character_lookup[128] {};
static bool         fen_character_lookup_initialised = false;
inline void         init_character_lookup() {

    if (fen_character_lookup_initialised)
        return;
    fen_character_lookup_initialised = true;

    fen_character_lookup['p']        = FenCharacter {'p', 1, BLACK_PAWN};
    fen_character_lookup['P']        = FenCharacter {'P', 1, WHITE_PAWN};
    fen_character_lookup['n']        = FenCharacter {'n', 1, BLACK_KNIGHT};
    fen_character_lookup['N']        = FenCharacter {'N', 1, WHITE_KNIGHT};
    fen_character_lookup['b']        = FenCharacter {'b', 1, BLACK_BISHOP, 1, 0};
    fen_character_lookup['B']        = FenCharacter {'B', 1, WHITE_BISHOP};
    fen_character_lookup['r']        = FenCharacter {'r', 1, BLACK_ROOK};
    fen_character_lookup['R']        = FenCharacter {'R', 1, WHITE_ROOK};
    fen_character_lookup['q']        = FenCharacter {'q', 1, BLACK_QUEEN};
    fen_character_lookup['Q']        = FenCharacter {'Q', 1, WHITE_QUEEN};
    fen_character_lookup['k']        = FenCharacter {'k', 1, BLACK_KING};
    fen_character_lookup['K']        = FenCharacter {'K', 1, WHITE_KING};

    fen_character_lookup['1']        = FenCharacter {'1', 1, NO_PIECE, 0, 0};
    fen_character_lookup['2']        = FenCharacter {'2', 2, NO_PIECE, 0, 1};
    fen_character_lookup['3']        = FenCharacter {'3', 3, NO_PIECE, 0, 2};
    fen_character_lookup['4']        = FenCharacter {'4', 4, NO_PIECE, 0, 3};
    fen_character_lookup['5']        = FenCharacter {'5', 5, NO_PIECE, 0, 4};
    fen_character_lookup['6']        = FenCharacter {'6', 6, NO_PIECE, 0, 5};
    fen_character_lookup['7']        = FenCharacter {'7', 7, NO_PIECE, 0, 6};
    fen_character_lookup['8']        = FenCharacter {'8', 8, NO_PIECE, 0, 7};

    fen_character_lookup['a']        = FenCharacter {'a', 1, NO_PIECE, 0, 0};
    //  fen_character_lookup['b'] = FenCharacter{'b', 2, NO_PIECE, 1, 0};
    fen_character_lookup['c'] = FenCharacter {'c', 3, NO_PIECE, 2, 0};
    fen_character_lookup['d'] = FenCharacter {'d', 4, NO_PIECE, 3, 0};
    fen_character_lookup['e'] = FenCharacter {'e', 5, NO_PIECE, 4, 0};
    fen_character_lookup['f'] = FenCharacter {'f', 6, NO_PIECE, 5, 0};
    fen_character_lookup['g'] = FenCharacter {'g', 7, NO_PIECE, 6, 0};
    fen_character_lookup['h'] = FenCharacter {'h', 8, NO_PIECE, 7, 0};

    fen_character_lookup['/'] = FenCharacter {'/', -16, NO_PIECE};
}

inline Position parse_fen(const std::string& fen) {

    init_character_lookup();

    // track which char of the fen we parse
    int character_index = 0;

    // the position itself
    Position position {};

    // -----------------------------------------------------------------------------------------------
    // read pieces first
    // -----------------------------------------------------------------------------------------------
    Square square = A8;
    Piece  pieces[64] {};
    std::memset(pieces, (Piece) NO_PIECE, sizeof(Piece) * 64);
    for (; character_index < fen.size() && fen[character_index] != ' '; character_index++) {
        FenCharacter& ch = fen_character_lookup[fen[character_index]];
        if (ch.piece != NO_PIECE) {
            pieces[square] = ch.piece;
        }
        square += ch.skip_squares;
    }
    // translate pieces to position format
    for (Square i = 0; i < 64; i++) {
        if (pieces[i] != NO_PIECE) {
            // add the corresponding piece to the given square
            int existing_pieces = position.piece_count();
            set(position.m_occupancy, i);
            position.m_pieces.set_piece(existing_pieces, pieces[i]);
        }
    }

    character_index++;

    // -----------------------------------------------------------------------------------------------
    // read active player
    // -----------------------------------------------------------------------------------------------
    if (fen[character_index] == 'w') {
        position.m_meta.set_stm(WHITE);
    } else {
        position.m_meta.set_stm(BLACK);
    }
    character_index += 2;

    // -----------------------------------------------------------------------------------------------
    // read castling rights
    // -----------------------------------------------------------------------------------------------
    for (; character_index < fen.size() && fen[character_index] != ' '; character_index++) {
        if (fen[character_index] == '-')
            continue;
        FenCharacter& ch    = fen_character_lookup[fen[character_index]];
        Side          side  = type_of(ch.piece) == QUEEN ? QUEEN_SIDE : KING_SIDE;
        Color         color = color_of(ch.piece);
        position.m_meta.set_castle(color, side, true);
    }
    character_index++;

    // -----------------------------------------------------------------------------------------------
    // read e.p. square
    // -----------------------------------------------------------------------------------------------
    if (fen[character_index] != '-') {
        Rank file = fen_character_lookup[fen[character_index++]].rank;
        Rank rank = fen_character_lookup[fen[character_index]].file;
        position.m_meta.set_ep_square(square_index(rank, file));
    }
    character_index += 2;

    // -----------------------------------------------------------------------------------------------
    // read 50 move rule
    // -----------------------------------------------------------------------------------------------
    if (fen[character_index] != '-') {
        auto string_length = fen.find_first_of(' ', character_index);
        auto numeric       = std::stoi(fen.substr(character_index, string_length));
        position.m_meta.set_fifty_mr(std::min(255, numeric));
        character_index = string_length - 1;
    }
    character_index += 2;

    // -----------------------------------------------------------------------------------------------
    // read move count
    // -----------------------------------------------------------------------------------------------
    if (fen[character_index] != '-') {
        auto string_length = fen.find_first_of(' ', character_index);
        auto numeric       = std::stoi(fen.substr(character_index, string_length));
        position.m_meta.set_move_count(std::min(255, numeric));
        character_index = string_length - 1;
    }
    character_index += 1;

    // -----------------------------------------------------------------------------------------------
    // read wdl and cp values
    // -----------------------------------------------------------------------------------------------
    auto left_bracket_pos  = fen.find_first_of('[', character_index);
    auto right_bracket_pos = fen.find_first_of(']', character_index);

    if (left_bracket_pos == std::string::npos || right_bracket_pos == std::string::npos) {
        return position;
    }

    auto    wdl             = std::stof(fen.substr(left_bracket_pos + 1, right_bracket_pos));
    auto    cp              = std::stof(fen.substr(right_bracket_pos + 1, fen.size()));

    int8_t  wdl_int         = std::round(wdl * 2 - 1);
    int16_t cp_int          = std::round(cp);

    position.m_result.score = cp_int;
    position.m_result.wdl   = wdl_int;

    return position;
}

inline std::string write_fen(const Position& position, bool write_score = false) {
    std::stringstream ss;

    // we do it in the same way we read a fen.
    // first, we write the pieces
    for (Rank n = 7; n >= 0; n--) {
        int counting = 0;
        for (File i = 0; i < 8; i++) {
            Square s     = square_index(n, i);

            int    piece = position.get_piece(s);
            if (piece == -1) {
                counting++;
            } else {
                if (counting != 0) {
                    ss << counting;
                }
                counting = 0;
                ss << piece_identifier[piece];
            }
        }
        if (counting != 0) {
            ss << counting;
        }
        if (n != 0)
            ss << "/";
    }

    // adding the active player (w for white, b for black) padded by spaces.
    ss << " ";
    ss << ((position.m_meta.stm() == WHITE) ? "w" : "b");
    ss << " ";

    // its relevant to add a '-' if no castling rights exist
    bool anyCastling = false;
    if (position.m_meta.can_castle(WHITE, QUEEN_SIDE)) {
        anyCastling = true;
        ss << "Q";
    }
    if (position.m_meta.can_castle(WHITE, KING_SIDE)) {
        anyCastling = true;
        ss << "K";
    }
    if (position.m_meta.can_castle(BLACK, QUEEN_SIDE)) {
        anyCastling = true;
        ss << "q";
    }
    if (position.m_meta.can_castle(BLACK, KING_SIDE)) {
        anyCastling = true;
        ss << "k";
    }
    // if not castling rights exist, add a '-' in order to be able to read the e.p. square.
    if (anyCastling == false) {
        ss << "-";
    }
    // similar to castling rights, we need to add a '-' if there is no e.p. square.
    if (position.m_meta.ep_square() >= 0 && position.m_meta.ep_square() < 64) {
        ss << " ";
        ss << square_identifier[position.m_meta.ep_square()];
    } else {
        ss << " -";
    }

    // we also add the fifty move counter and the move counter to the fen (note that we dont parse
    // those)
    ss << " " << (int) position.m_meta.fifty_mr();
    ss << " " << (int) position.m_meta.move_count();

    if (write_score) {
        ss << " [";
        ss << (position.m_result.wdl == WIN ? "1" : (position.m_result.wdl == LOSS ? "0" : "0.5"));
        ss << "] " << position.m_result.score;
    }

    return ss.str();
}
}    // namespace chess
