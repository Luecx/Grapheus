//
// Created by finne on 07.04.2023.
//

#pragma once
namespace nn {

struct SplitHead : public Layer {

    size_t offset;
    Layer* prev;

    explicit SplitHead(Layer* prev, size_t size, size_t offset)
        : prev {prev}
        , Layer(size)
        , offset(offset) {}

    void compile(size_t batch_size) override {
        this->dense_output = Tape {prev->dense_output, size, batch_size, offset, 0};
    }
};

struct Split : public Layer {
    SplitHead heads[2];

    explicit Split(Layer* prev, size_t head_1_size)
        : Layer(prev->size)
        , heads {{SplitHead(prev, head_1_size, 0)},
                 {SplitHead(prev, prev->size - head_1_size, head_1_size)}} {}

    SplitHead* operator[](size_t s) {
        return &heads[s];
    }

    void compile(size_t batch_size) override {
        heads[0].compile(batch_size);
        heads[1].compile(batch_size);
    }
};

}    // namespace nn

