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

//// TODO: more than 2 heads
//struct Split : public Layer {
//    SplitHead heads[2];
//
//    explicit Split(Layer* prev, size_t head_1_size)
//        : Layer(prev->size)
//        , heads {{SplitHead(prev, head_1_size, 0)},
//                 {SplitHead(prev, prev->size - head_1_size, head_1_size)}} {
//        prev->use();
//    }
//
//    SplitHead* operator[](size_t s) {
//        return &heads[s];
//    }
//
//    void compile(size_t batch_size) override {
//        heads[0].compile(batch_size);
//        heads[1].compile(batch_size);
//    }
//};

struct Split : public Layer {
    std::vector<SplitHead> heads;

    explicit Split(Layer* prev, const std::vector<size_t>& head_sizes)
        : Layer(prev->size) {
        size_t total_assigned_size = 0;
        for (size_t i = 0; i < head_sizes.size(); ++i) {
            heads.emplace_back(prev, head_sizes[i], total_assigned_size);
            total_assigned_size += head_sizes[i];
        }
        // Now calculate and create the last head with the remaining size
        size_t remaining_size = prev->size - total_assigned_size;
        heads.emplace_back(prev, remaining_size, total_assigned_size);

        prev->use();
    }

    SplitHead* operator[](size_t index) {
        if (index < heads.size()) {
            return &heads[index];
        }
        throw std::out_of_range("Index out of range for Split heads");
    }

    void compile(size_t batch_size) override {
        for (auto& head : heads) {
            head.compile(batch_size);
        }
    }
};

}    // namespace nn

