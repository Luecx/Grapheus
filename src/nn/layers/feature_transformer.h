#pragma once

namespace nn {

struct FeatureTransformer : public Layer {

    SparseInput* inp1;
    SparseInput* inp2;

    Tape         weights {0, 0};
    Tape         bias {0, 0};

    private:
    Tape out_1{0,0};
    Tape out_2{0,0};

    public:
    FeatureTransformer(SparseInput* inp1, SparseInput* inp2, size_t half_size)
        : Layer(2 * half_size)
        , inp1(inp1)
        , inp2(inp2) {
        inp1->use();
        inp2->use();

        ERROR(inp1->size == inp2->size);

        weights = Tape(size / 2, inp1->size);
        weights.malloc();
        math::normal(weights.values,
                     0.f,
                     1.0f / std::sqrtf(inp1->max_inputs));
        weights.values >> data::GPU;

        bias = Tape(size / 2, 1);
        bias.malloc();
    }

    void compile(size_t batch_size) override {
        // set output matrix
        compile_suboutput(batch_size, Tape(size, batch_size));
    }

    void compile_suboutput(size_t batch_size, const Tape& output) override {
        Layer::compile_suboutput(batch_size, output);

        out_1 = Tape(this->dense_output, this->size / 2, batch_size, 0, 0);
        out_2 = Tape(this->dense_output, this->size / 2, batch_size, this->size / 2, 0);
    }

    void forward() override {
        operations::affine_sparse<data::GPU>(
            weights.values, inp1->sparse_output, bias.values, out_1.values);
        operations::affine_sparse<data::GPU>(
            weights.values, inp2->sparse_output, bias.values, out_2.values);
    }
    void backward() override {
        operations::affine_sparse_bp<data::GPU>(
            weights.gradients, inp1->sparse_output, bias.gradients, out_1.gradients);
        operations::affine_sparse_bp<data::GPU>(
            weights.gradients, inp2->sparse_output, bias.gradients, out_2.gradients);
    }

    std::vector<Tape*> params() override {
        return std::vector<Tape*>{&weights, &bias};
    }
};

}    // namespace nn