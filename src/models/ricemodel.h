#pragma once

#include "chessmodelbinpack.h"

namespace model {
struct RiceModel : ChessModelBinpack {
    SparseInput* in1;
    SparseInput* in2;

    int          threads = 6;

    // clang-format off
    // King bucket indicies
    static constexpr int indices[64] = {
        0,  1,  2,  3,  3,  2,  1,  0,
        4,  5,  6,  7,  7,  6,  5,  4,
        8,  9,  10, 11, 11, 10, 9,  8,
        12, 13, 14, 15, 15, 14, 13, 12,
        16, 17, 18, 19, 19, 18, 17, 16,
        20, 21, 22, 23, 23, 22, 21, 20,
        24, 25, 26, 27, 27, 26, 25, 24,
        28, 29, 30, 31, 31, 30, 29, 28,
    };

    // static constexpr int indices[64] = {
    //     0,  1,  2,  3,  3,  2,  1,  0,
    //     4,  5,  6,  7,  7,  6,  5,  4,
    //     8,  9,  10, 11, 11, 10, 9,  8,
    //     8,  9,  10, 11, 11, 10, 9,  8,
    //     12, 12, 13, 13, 13, 13, 12, 12,
    //     12, 12, 13, 13, 13, 13, 12, 12,
    //     14, 14, 15, 15, 15, 15, 14, 14,
    //     14, 14, 15, 15, 15, 15, 14, 14,
    // };
    // clang-format on

    RiceModel(size_t n_ft, float start_lambda, float end_lambda, size_t save_rate)
        : ChessModelBinpack(start_lambda, end_lambda) {
        in1     = add<SparseInput>(12 * 64 * 32, 32);
        in2     = add<SparseInput>(12 * 64 * 32, 32);

        auto ft = add<FeatureTransformer>(in1, in2, n_ft);
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

        add_quantization(Quantizer {
            "quant",
            save_rate,
            QuantizerEntry<int16_t>(&ft->weights.values, 32, true),
            QuantizerEntry<int16_t>(&ft->bias.values, 32),
            QuantizerEntry<int16_t>(&af->weights.values, 128),
            QuantizerEntry<int32_t>(&af->bias.values, 32 * 128),
        });
        set_save_frequency(save_rate);
    }

    inline int king_square_index(int kingSquare, uint8_t kingColor) {
        kingSquare = (56 * kingColor) ^ kingSquare;
        return indices[kingSquare];
    }

    inline int
        index(uint8_t pieceType, uint8_t pieceColor, int square, uint8_t view, int kingSquare) {
        const int ksIndex = king_square_index(kingSquare, view);
        square            = square ^ (56 * view);
        square            = square ^ (7 * !!(kingSquare & 0x4));

        // clang-format off
        return square
            + pieceType * 64
            + !(pieceColor ^ view) * 64 * 6 + ksIndex * 64 * 6 * 2;
        // clang-format on
    }

    void setup_inputs_and_outputs(binpackloader::DataSet& positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static, 64) num_threads(threads)
        for (int b = 0; b < positions.size(); b++) {
            const auto pos = positions[b].pos;
            // fill in the inputs and target values

            const auto wKingSq = pos.kingSquare(binpack::chess::Color::White);
            const auto bKingSq = pos.kingSquare(binpack::chess::Color::Black);

            const auto pieces  = pos.piecesBB();

            for (auto sq : pieces) {
                const auto         piece                 = pos.pieceAt(sq);
                const std::uint8_t pieceType             = static_cast<uint8_t>(piece.type());
                const std::uint8_t pieceColor            = static_cast<uint8_t>(piece.color());

                auto               piece_index_white_pov = index(pieceType,
                                                   pieceColor,
                                                   static_cast<int>(sq),
                                                   static_cast<uint8_t>(binpack::chess::Color::White),
                                                   static_cast<int>(wKingSq));
                auto               piece_index_black_pov = index(pieceType,
                                                   pieceColor,
                                                   static_cast<int>(sq),
                                                   static_cast<uint8_t>(binpack::chess::Color::Black),
                                                   static_cast<int>(bKingSq));

                if (pos.sideToMove() == binpack::chess::Color::White) {
                    in1->sparse_output.set(b, piece_index_white_pov);
                    in2->sparse_output.set(b, piece_index_black_pov);
                } else {
                    in2->sparse_output.set(b, piece_index_white_pov);
                    in1->sparse_output.set(b, piece_index_black_pov);
                }
            }

            float p_value  = positions[b].score;
            float w_value  = positions[b].result;

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            float actual_lambda =
                start_lambda + (end_lambda - start_lambda) * (current_epoch / max_epochs);

            target(b) = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target) / 1.0f;
        }
    }

    void setup_inputs_and_outputs(binpackloader::DataEntry& entry) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto&      target = m_loss->target;

        const auto pos    = entry.pos;
        // fill in the inputs and target values

        const auto wKingSq = pos.kingSquare(binpack::chess::Color::White);
        const auto bKingSq = pos.kingSquare(binpack::chess::Color::Black);

        const auto pieces  = pos.piecesBB();

        for (auto sq : pieces) {
            const auto         piece                 = pos.pieceAt(sq);
            const std::uint8_t pieceType             = static_cast<uint8_t>(piece.type());
            const std::uint8_t pieceColor            = static_cast<uint8_t>(piece.color());

            auto               piece_index_white_pov = index(pieceType,
                                               pieceColor,
                                               static_cast<int>(sq),
                                               static_cast<uint8_t>(binpack::chess::Color::White),
                                               static_cast<int>(wKingSq));
            auto               piece_index_black_pov = index(pieceType,
                                               pieceColor,
                                               static_cast<int>(sq),
                                               static_cast<uint8_t>(binpack::chess::Color::Black),
                                               static_cast<int>(bKingSq));

            if (pos.sideToMove() == binpack::chess::Color::White) {
                in1->sparse_output.set(0, piece_index_white_pov);
                in2->sparse_output.set(0, piece_index_black_pov);
            } else {
                in2->sparse_output.set(0, piece_index_white_pov);
                in1->sparse_output.set(0, piece_index_black_pov);
            }
        }
    }
};
}    // namespace model