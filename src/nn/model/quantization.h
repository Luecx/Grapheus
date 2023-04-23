#pragma once

#include <type_traits>
#include <utility>

namespace nn {

struct QuantizerEntryBase {
    virtual ~QuantizerEntryBase() {}
    virtual void write(std::ostream& output) = 0;
};

template<typename TYPE>
struct QuantizerEntry : QuantizerEntryBase {
    data::DenseMatrix<float>* m_reference;
    float                     m_scalar;
    bool                      m_transpose;

    QuantizerEntry(data::DenseMatrix<float>* ref, float scalar, bool transpose = false)
        : m_reference(ref)
        , m_scalar(scalar)
        , m_transpose(transpose) {}

    void write(std::ostream& output) override {
        int m = m_reference->m;
        int n = m_reference->n;

        if (m_transpose) {
            m = m_reference->n;
            n = m_reference->m;
        }

        *m_reference >> data::CPU;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float value;
                if (m_transpose) {
                    value = m_reference->get(j, i) * m_scalar;
                } else {
                    value = m_reference->get(i, j) * m_scalar;
                }
                if constexpr (std::is_floating_point_v<TYPE>) {
                    output.write(reinterpret_cast<char*>(&value), sizeof(value));
                } else {
                    TYPE quantized_value;
                    if (value < std::numeric_limits<TYPE>::lowest()) {
                        quantized_value = std::numeric_limits<TYPE>::lowest();
                    } else if (value > std::numeric_limits<TYPE>::max()) {
                        quantized_value = std::numeric_limits<TYPE>::max();
                    } else {
                        quantized_value = static_cast<TYPE>(std::round(value));
                    }
                    output.write(reinterpret_cast<char*>(&quantized_value), sizeof(quantized_value));
                }
            }
        }
    }
};

struct Quantizer {
    std::vector<std::shared_ptr<QuantizerEntryBase>> entries;
    std::string                                      name;
    size_t                                           frequency;

    std::filesystem::path                            path;

    template<typename... TYPES>
    Quantizer(std::string name, size_t frequency, QuantizerEntry<TYPES>... args)
        : name(std::move(name))
        , frequency(frequency) {
        (entries.push_back(std::make_shared<QuantizerEntry<TYPES>>(args)), ...);
    }

    void set_path(const std::filesystem::path& base_path) {
        path = base_path / name;
        std::filesystem::create_directories(path);
    }

    void save(int epoch) {
        if (epoch % frequency == 0) {
            save("epoch_" + std::to_string(epoch) + ".net");
        }
    }

    void save(std::string name) {
        std::ofstream file(path / name, std::ios::out | std::ios::binary);

        for (auto& entry : entries) {
            entry->write(file);
        }

        file.close();
    }
};
}