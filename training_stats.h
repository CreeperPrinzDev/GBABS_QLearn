#pragma once

#include "hitpot.h"

#include <array>
#include <cstdint>
#include <iostream>

struct TrainingStats {
    uint64_t samples = 0;
    uint64_t totalReward = 0;

    std::array<uint64_t, 7> states{};

    void add(const HitPotResult& r) {
        samples++;
        totalReward += r.reward;

        switch (r.state) {
        case HitPotState::POT_HIT:     states[0]++; break;
        case HitPotState::VERY_HOT:    states[1]++; break;
        case HitPotState::HOT:         states[2]++; break;
        case HitPotState::MEDIUM:      states[3]++; break;
        case HitPotState::COLD:        states[4]++; break;
        case HitPotState::BLIZZARD:    states[5]++; break;
        case HitPotState::ZERO_KELVIN: states[6]++; break;
        }
    }

    double accuracyPercent() const {
        if (samples == 0) return 0.0;
        return (double(totalReward) / double(samples * 100ULL)) * 100.0;
    }

    void print() const {
        std::cout << "Samples: " << samples << "\n";
        std::cout << "Accuracy: " << accuracyPercent() << "%\n";
        std::cout << "POT_HIT: " << states[0] << "\n";
        std::cout << "VERY_HOT: " << states[1] << "\n";
        std::cout << "HOT: " << states[2] << "\n";
        std::cout << "MEDIUM: " << states[3] << "\n";
        std::cout << "COLD: " << states[4] << "\n";
        std::cout << "BLIZZARD: " << states[5] << "\n";
        std::cout << "ZERO_KELVIN: " << states[6] << "\n";

        if (accuracyPercent() > 95.0) {
            std::cout << "[SUS] Accuracy > 95%. Check for leakage/overfitting.\n";
        }
    }
};
