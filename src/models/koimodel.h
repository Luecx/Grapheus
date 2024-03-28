#pragma once

#include "chessmodel.h"

namespace model {

struct KoiModel : ChessModel {
    SparseInput* in1;
    SparseInput* in2;

    KoiModel(float lambda, size_t save_rate)
        : ChessModel(lambda) {
        in1     = add<SparseInput>(16 * 12 * 64, 32);
        in2     = add<SparseInput>(16 * 12 * 64, 32);

        auto ft = add<FeatureTransformer>(in1, in2, 512);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re, 1);
        auto sm = add<Sigmoid>(af, 2.5 / 400);

        add_optimizer(Adam({{OptimizerEntry {&ft->weights}},
                            {OptimizerEntry {&ft->bias}},
                            {OptimizerEntry {&af->weights}},
                            {OptimizerEntry {&af->bias}}},
                           0.9,
                           0.999,
                           1e-8));

        set_save_frequency(save_rate);
        add_quantization(Quantizer {
            "quant_1",
            10,
            QuantizerEntry<int16_t>(&ft->weights.values, 32, true),
            QuantizerEntry<int16_t>(&ft->bias.values, 32),
            QuantizerEntry<int16_t>(&af->weights.values, 128),
            QuantizerEntry<int32_t>(&af->bias.values, 128 * 32),
        });
    }

    inline int king_square_index(chess::Square relative_king_square) {

        // clang-format off
        constexpr int indices[chess::N_SQUARES] {
            0,  1,  2,  3,  3,  2,  1,  0,
            4,  5,  6,  7,  7,  6,  5,  4,
            8,  9,  10, 11, 11, 10, 9,  8,
            8,  9,  10, 11, 11, 10, 9,  8,
            12, 12, 13, 13, 13, 13, 12, 12,
            12, 12, 13, 13, 13, 13, 12, 12,
            14, 14, 15, 15, 15, 15, 14, 14,
            14, 14, 15, 15, 15, 15, 14, 14,
        };
        // clang-format on

        return indices[relative_king_square];
    }

    inline int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {
        constexpr int          PIECE_TYPE_FACTOR  = 64;
        constexpr int          PIECE_COLOR_FACTOR = PIECE_TYPE_FACTOR * 6;
        constexpr int          KING_SQUARE_FACTOR = PIECE_COLOR_FACTOR * 2;

        const chess::PieceType piece_type         = chess::type_of(piece);
        const chess::Color     piece_color        = chess::color_of(piece);

        chess::Square          relative_king_square;
        chess::Square          relative_piece_square;

        if (view == chess::WHITE) {
            relative_king_square  = king_square;
            relative_piece_square = piece_square;
        } else {
            relative_king_square  = chess::mirror_ver(king_square);
            relative_piece_square = chess::mirror_ver(piece_square);
        }

        const int king_square_idx = king_square_index(relative_king_square);
        if (chess::file_index(king_square) > 3) {
            relative_piece_square = chess::mirror_hor(relative_piece_square);
        }

        const int index = relative_piece_square + piece_type * PIECE_TYPE_FACTOR
                          + (piece_color == view) * PIECE_COLOR_FACTOR
                          + king_square_idx * KING_SQUARE_FACTOR;
        return index;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(16)
        for (int b = 0; b < positions->header.entry_count; b++) {
            chess::Position* pos = &positions->positions[b];
            // fill in the inputs and target values

            chess::Square wKingSq = pos->get_king_square<chess::WHITE>();
            chess::Square bKingSq = pos->get_king_square<chess::BLACK>();

            chess::BB     bb {pos->m_occupancy};
            int           idx = 0;

            while (bb) {
                chess::Square sq                    = chess::lsb(bb);
                chess::Piece  pc                    = pos->m_pieces.get_piece(idx);

                auto          piece_index_white_pov = index(sq, pc, wKingSq, chess::WHITE);
                auto          piece_index_black_pov = index(sq, pc, bKingSq, chess::BLACK);

                if (pos->m_meta.stm() == chess::WHITE) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }

                bb = chess::lsb_reset(bb);
                idx++;
            }

            float p_value = pos->m_result.score;
            float w_value = pos->m_result.wdl;

            // flip if black is to move -> relative network style
            if (pos->m_meta.stm() == chess::BLACK) {
                p_value = -p_value;
                w_value = -w_value;
            }

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            target(b)      = lambda * p_target + (1.0 - lambda) * w_target;
        }
    }
};
}    // namespace model