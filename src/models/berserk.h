#pragma once

#include "chessmodel.h"

namespace model {

struct BerserkModel : ChessModel<dataset::BatchLoader<chess::Position>> {
    using DataLoader = dataset::BatchLoader<chess::Position>;

    SparseInput* in1;
    SparseInput* in2;

    const float  sigmoid_scale = 1.0 / 160.0;
    const float  quant_one     = 32.0;
    const float  quant_two     = 32.0;

    const size_t n_features    = 16 * 12 * 64;
    const size_t n_l1          = 16;
    const size_t n_l2          = 32;
    const size_t n_out         = 1;

    float        lambda        = 0.5;

    BerserkModel(DataLoader&                train_loader,
                 std::optional<DataLoader>& val_loader,
                 size_t                     n_ft,
                 float                      lambda,
                 size_t                     save_rate)
        : ChessModel(train_loader, val_loader)
        , lambda(lambda) {

        in1                    = add<SparseInput>(n_features, 32);
        in2                    = add<SparseInput>(n_features, 32);

        auto ft                = add<FeatureTransformer>(in1, in2, n_ft);
        auto fta               = add<ClippedRelu>(ft);
        ft->ft_regularization  = 1.0 / 16384.0 / 4194304.0;
        fta->max               = 127.0;

        auto        l1         = add<Affine>(fta, n_l1);
        auto        l1a        = add<ReLU>(l1);

        auto        l2         = add<Affine>(l1a, n_l2);
        auto        l2a        = add<ReLU>(l2);

        auto        pos_eval   = add<Affine>(l2a, n_out);
        auto        sigmoid    = add<Sigmoid>(pos_eval, sigmoid_scale);

        const float hidden_max = 127.0 / quant_two;
        add_optimizer(AdamWarmup({{OptimizerEntry {&ft->weights}},
                                  {OptimizerEntry {&ft->bias}},
                                  {OptimizerEntry {&l1->weights}.clamp(-hidden_max, hidden_max)},
                                  {OptimizerEntry {&l1->bias}},
                                  {OptimizerEntry {&l2->weights}},
                                  {OptimizerEntry {&l2->bias}},
                                  {OptimizerEntry {&pos_eval->weights}},
                                  {OptimizerEntry {&pos_eval->bias}}},
                                 0.95,
                                 0.999,
                                 1e-8,
                                 5 * 16384));

        set_save_frequency(save_rate);
        add_quantization(Quantizer {
            "quant",
            save_rate,
            QuantizerEntry<int16_t>(&ft->weights.values, quant_one, true),
            QuantizerEntry<int16_t>(&ft->bias.values, quant_one),
            QuantizerEntry<int8_t>(&l1->weights.values, quant_two),
            QuantizerEntry<int32_t>(&l1->bias.values, quant_two),
            QuantizerEntry<float>(&l2->weights.values, 1.0),
            QuantizerEntry<float>(&l2->bias.values, quant_two),
            QuantizerEntry<float>(&pos_eval->weights.values, 1.0),
            QuantizerEntry<float>(&pos_eval->bias.values, quant_two),
        });
    }

    static inline int king_square_index(int relative_king_square) {
        constexpr int indices[64] {
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 14, 14, 15, 15,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 12, 12, 13, 13,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 8,  9,  10, 11,    //
            -1, -1, -1, -1, 4,  5,  6,  7,     //
            -1, -1, -1, -1, 0,  1,  2,  3,     //
        };

        return indices[relative_king_square];
    }

    static inline int index(chess::Square piece_square,
                            chess::Piece  piece,
                            chess::Square king_square,
                            chess::Color  view) {

        const chess::PieceType piece_type  = chess::type_of(piece);
        const chess::Color     piece_color = chess::color_of(piece);

        piece_square ^= 56;
        king_square ^= 56;

        const int oP  = piece_type + 6 * (piece_color != view);
        const int oK  = (7 * !(king_square & 4)) ^ (56 * view) ^ king_square;
        const int oSq = (7 * !(king_square & 4)) ^ (56 * view) ^ piece_square;

        return king_square_index(oK) * 12 * 64 + oP * 64 + oSq;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>& positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(6)
        for (int b = 0; b < positions.header.entry_count; b++) {
            auto& pos = positions.positions[b];

            DataLoader::set_features(b, pos, in1, in2, index);

            float p_value = DataLoader::get_p_value(pos);
            float w_value = DataLoader::get_w_value(pos);

            // flip if black is to move -> relative network style
            if (pos.m_meta.stm() == chess::BLACK) {
                p_value = -p_value;
                w_value = -w_value;
            }

            float p_target = 1 / (1 + expf(-p_value * sigmoid_scale));
            float w_target = (w_value + 1) / 2.0f;

            target(b)      = lambda * p_target + (1.0 - lambda) * w_target;

            // layer_selector->dense_output.values(b, 0) =
            //     (int) ((chess::popcount(pos->m_occupancy) - 1) / 4);
        }
    }

    void setup_inputs_and_outputs_only(chess::Position& pos) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        DataLoader::set_features(0, pos, in1, in2, index);
    }
};

}    // namespace model