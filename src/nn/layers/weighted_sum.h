#pragma once
namespace nn {
struct WeightedSum : public Layer {

    Layer* prev_1;
    Layer* prev_2;

    float  alpha;
    float  beta;

    WeightedSum(Layer* prev1, Layer* prev2, float alpha, float beta)
        : Layer(prev1->size)
        , prev_1(prev1)
        , prev_2(prev2)
        , alpha(alpha)
        , beta(beta) {
        prev1->use();
        prev2->use();
        ERROR(prev1->size == prev2->size);
    }

    void compile(size_t batch_size) override {
        this->compile_suboutput(batch_size, Tape(size, batch_size));
    }

    void forward() override {
        operations::ax_p_by<data::GPU>(prev_1->dense_output.values,
                                       prev_2->dense_output.values,
                                       dense_output.values,
                                       alpha,
                                       beta);
    }
    void backward() override {
        operations::ax_p_by_bp<data::GPU>(prev_1->dense_output.gradients,
                                          prev_2->dense_output.gradients,
                                          dense_output.gradients,
                                          alpha,
                                          beta);
    }
};

}    // namespace nn
