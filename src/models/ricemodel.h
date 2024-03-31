#pragma once

#include "chessmodel.h"

namespace model {
struct RiceModel : ChessModel<binpackloader::BinpackLoader> {
    using DataLoader = binpackloader::BinpackLoader;

    SparseInput* in1;
    SparseInput* in2;

    int          threads      = 6;
    float        start_lambda = 0.7;
    float        end_lambda   = 0.7;

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
    // clang-format on

    RiceModel(binpackloader::BinpackLoader&                train_loader,
              std::optional<binpackloader::BinpackLoader>& val_loader,
              size_t                                       n_ft,
              float                                        start_lambda,
              float                                        end_lambda,
              size_t                                       save_rate)

        : start_lambda(start_lambda)
        , end_lambda(end_lambda)
        , ChessModel(train_loader, val_loader) {

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

    static int king_square_index(int kingSquare, uint8_t kingColor) {
        kingSquare = (56 * kingColor) ^ kingSquare;
        return indices[kingSquare];
    }

    static int
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
            const auto& entry = positions[b];

            DataLoader::set_features(b, entry, in1, in2, index);

            float p_value  = DataLoader::get_p_value(positions[b]);
            float w_value  = DataLoader::get_w_value(positions[b]);

            float p_target = 1 / (1 + expf(-p_value * 2.5 / 400));
            float w_target = (w_value + 1) / 2.0f;

            float actual_lambda =
                start_lambda + (end_lambda - start_lambda) * (current_epoch / max_epochs);

            target(b) = (actual_lambda * p_target + (1.0f - actual_lambda) * w_target) / 1.0f;
        }
    }

    void setup_inputs_and_outputs_only(binpackloader::DataEntry& entry) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        DataLoader::set_features(0, entry, in1, in2, index);
    }
};
}    // namespace model