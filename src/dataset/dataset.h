#pragma once

#include "../math/random.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dataset {

struct DataSetEntry {};

struct DataSetHeader {
    uint64_t entry_count;

    char     label_1[128];
    char     label_2[128];
    char     label_3[1024];
};

template<typename TYPE>
struct DataSet {
    static_assert(std::is_base_of<DataSetEntry, TYPE>::value,
                  "TYPE must be a subclass of DataSetEntry");
    DataSetHeader     header {};
    std::vector<TYPE> positions {};

    // adds another dataset to this one
    void addData(DataSet<TYPE>& other) {
        positions.insert(std::end(positions), std::begin(other.positions), std::end(other.positions));
        header.entry_count += other.header.entry_count;
    }

    void resize(size_t size) {
        positions.resize(size);
        header.entry_count = size;
    }

    // shuffle
    void shuffle() {
        std::shuffle(positions.begin(), positions.end(), math::twister);
    }
};

}    // namespace dataset
