#pragma once

#include <cmath>

namespace nn {
/**
 * @brief Base class for learning rate schedules.
 *
 * This class defines the interface for all learning rate schedules.
 * Subclasses should implement the `get_lr` method to return the learning rate
 * for a given epoch.
 */
struct LRSchedule {

    /**
     * @brief Constructor for LRSchedule.
     *
     * @param learning_rate The initial learning rate.
     */
    LRSchedule(float learning_rate)
        : initial_lr(learning_rate) {}

    protected:
    float initial_lr;    // initial learning rate

    public:
    /**
     * @brief Get the learning rate for a given epoch.
     *
     * @param epoch The current epoch.
     * @return The learning rate for the given epoch.
     */
    virtual float get_lr(int epoch) const = 0;
};

/**
 * @brief Learning rate schedule with a fixed learning rate.
 *
 * This schedule returns a fixed learning rate for all epochs.
 */
struct FixedLRSchedule : public LRSchedule {
    /**
     * @brief Constructor for FixedLRSchedule.
     *
     * @param learning_rate The fixed learning rate.
     */
    FixedLRSchedule(float learning_rate)
        : LRSchedule(learning_rate) {}

    protected:
    virtual float get_lr(int epoch) const override {
        return initial_lr;
    }
};

/**
 * @brief Learning rate schedule with step decay.
 *
 * This schedule decays the learning rate by a fixed factor after a certain
 * number of epochs.
 */
struct StepDecayLRSchedule : public LRSchedule {
    /**
     * @brief Constructor for StepDecayLRSchedule.
     *
     * @param learning_rate The initial learning rate.
     * @param decay_rate_ The decay rate for the learning rate.
     * @param decay_steps_ The number of epochs between decay steps.
     */
    StepDecayLRSchedule(float learning_rate, float decay_rate_, int decay_steps_)
        : LRSchedule(learning_rate)
        , decay_rate(decay_rate_)
        , decay_steps(decay_steps_) {}

    protected:
    float         decay_rate;     // The decay rate for the learning rate.
    int           decay_steps;    // The number of epochs between decay steps.
    virtual float get_lr(int epoch) const override {
        return initial_lr * pow(decay_rate, std::floor((float) (epoch - 1) / decay_steps));
    }
};

/**
 * @brief Learning rate schedule with cosine annealing.
 *
 * This schedule uses cosine annealing to smoothly reduce the learning rate
 * over a given number of epochs.
 */
struct CosineAnnealingLRSchedule : public LRSchedule {
    /**
     * @brief Constructor for CosineAnnealingLRSchedule.
     *
     * @param learning_rate The initial learning rate.
     * @param T_max_ The maximum number of epochs.
     */
    CosineAnnealingLRSchedule(float learning_rate, int ep_max)
        : LRSchedule(learning_rate)
        , ep_max(ep_max) {}

    protected:
    int           ep_max;    // maximum number of epochs.

    virtual float get_lr(int epoch) const override {
        return 0.5f * initial_lr * (1 + std::cos(3.1415926535f * epoch / ep_max));
    }
};
}    // namespace nn
