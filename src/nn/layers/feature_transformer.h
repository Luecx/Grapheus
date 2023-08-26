#pragma once

namespace nn {

struct FeatureTransformer : public Layer {

    SparseInput* inp1;
    SparseInput* inp2;

    Tape         weights {0, 0};
    Tape         bias {0, 0};

    float ft_regularization = 0.0;

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
        bias = Tape(size / 2, 1);
        bias.malloc();

        math::kaiming<float>(weights.values, inp1->max_inputs);
        math::fill<float>(bias.values, 0.0f);

        weights.values >> data::GPU;
        bias   .values >> data::GPU;
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
//        operations::affine_sparse_shared<data::GPU>(weights.values, inp1->sparse_output, inp2->sparse_output, bias.values, dense_output.values);
    }
    void backward() override {
        operations::affine_sparse_bp<data::GPU>(
            weights.gradients, inp1->sparse_output, bias.gradients, out_1.values, out_1.gradients, ft_regularization);
        operations::affine_sparse_bp<data::GPU>(
            weights.gradients, inp2->sparse_output, bias.gradients, out_2.values, out_2.gradients, ft_regularization);
        // dont use this code, its slower for some stupid reason
//        operations::affine_sparse_shared_bp<data::GPU>(weights.gradients, inp1->sparse_output, inp2->sparse_output, bias.gradients, dense_output.gradients);
    }

    std::vector<Tape*> params() override {
        return std::vector<Tape*>{&weights, &bias};
    }
};

}    // namespace nn