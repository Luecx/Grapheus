#pragma once

#include "chessmodel.h"

namespace model {

struct KPModel : ChessModel {

    SparseInput* in1;
    SparseInput* in2;

    DenseInput*  in_big;
    DenseInput*  in_offset;
    DenseInput*  in_one;

    const float in_scaling = 340;
    const float out_scaling = 380;
    const float offset = 270;

    KPModel(size_t n_ft, float lambda, size_t save_rate)
        : ChessModel(lambda) {

        in1 = add<SparseInput>(4 * 64 * 64, 32);
        in2 = add<SparseInput>(4 * 64 * 64, 32);

        auto ft = add<FeatureTransformer>(in1, in2, 512);
        auto re = add<ReLU>(ft);
        auto af = add<Affine>(re, 1);

        // setting the input of the main network with exactly one neuron
        in_big    = add<DenseInput>(1);
        in_offset = add<DenseInput>(1);
        in_one    = add<DenseInput>(1);

        // network eval (scorenet)
        auto scorenet = add<WeightedSum>(in_big, af, 1, 1);
        // q  = ( scorenet - offset) / in_scaling
        // qm = (-scorenet - offset) / in_scaling
        auto q   = add<WeightedSum>(scorenet, in_offset,   1 / in_scaling, -1 / in_scaling);
        auto qm  = add<WeightedSum>(scorenet, in_offset, - 1 / in_scaling, -1 / in_scaling);
        auto qs1 = add<Sigmoid>(q,  1.0);
        auto qs2 = add<Sigmoid>(qm, 1.0);
        // qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())
        auto qfa = add<WeightedSum>(in_one, qs1, 1.0, 1.0);
        auto qf  = add<WeightedSum>(qfa, qs2, 0.5, -0.5);

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

    inline int index(chess::Square piece_square,
                     chess::Piece  piece,
                     chess::Square king_square,
                     chess::Color  view) {

        const chess::PieceType piece_type  = chess::type_of(piece);
        const chess::Color     piece_color = chess::color_of(piece);

        // flip if view is black
        king_square  ^= (56 * view);
        piece_square ^= (56 * view);

        // get the idx, 0 for pawn, 1 for king
        int piece_idx = piece_type == chess::PAWN ? 0 : 1;

        // if the piece color is not the view, add two to the piece index
        if (piece_color != view) {
            piece_idx += 2;
        }

        return piece_idx   * 64 * 64 +
               king_square * 64 +
               piece_square;
    }

    void setup_inputs_and_outputs(dataset::DataSet<chess::Position>* positions) {
        in1->sparse_output.clear();
        in2->sparse_output.clear();

        auto& target = m_loss->target;

#pragma omp parallel for schedule(static) num_threads(6)
        for (int b = 0; b < positions->header.entry_count; b++) {
            chess::Position* pos = &positions->positions[b];

            // -----------------------------------------------------
            // compute the inputs to the network
            // -----------------------------------------------------
            chess::Square wKingSq = pos->get_king_square<chess::WHITE>();
            chess::Square bKingSq = pos->get_king_square<chess::BLACK>();

            chess::BB     bb {pos->m_occupancy};
            int           idx = 0;

            while (bb) {
                chess::Square sq                    = chess::lsb(bb);
                chess::Piece  pc                    = pos->m_pieces.get_piece(idx);

                auto          piece_index_white_pov = index(sq, pc, wKingSq, chess::WHITE);
                auto          piece_index_black_pov = index(sq, pc, bKingSq, chess::BLACK);

                if (pc != chess::PAWN || pc != chess::KING) {
                    continue;
                }

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


            // -----------------------------------------------------
            // compute the target values for the network
            // -----------------------------------------------------
            float p_value = pos->m_result.score();
            float w_value = pos->m_result.wdl();

            // flip if black is to move -> relative network style
            if (pos->m_meta.stm() == chess::BLACK) {
                // p_value = -p_value;  <-- score is already relative from pov
                w_value = -w_value;
            }

            float p  = ( p_value - offset) / out_scaling;
            float pm = (-p_value - offset) / out_scaling;
            float pf = 0.5 * (1.0 + (1) / (1 + std::exp(-p)) - (1) / (1 + std::exp(-pm)));

            float p_target = 1 / (1 + expf(-p_value * 2.5f / 400.0f));
            float w_target = (w_value + 1) / 2.0f;

            target(b)      = lambda * p_target + (1.0 - lambda) * w_target;
        }
    }
};

}