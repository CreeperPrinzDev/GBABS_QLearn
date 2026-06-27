#pragma once
#include <cstdint>
#include <string>

enum class HitPotState {
    POT_HIT,
    VERY_HOT,
    HOT,
    MEDIUM,
    COLD,
    BLIZZARD,
    ZERO_KELVIN
};

struct HitPotResult {
    HitPotState state;
    int reward;
    int distance;
};

HitPotResult scoreSymbol(uint8_t teacher, uint8_t student);
std::string hitPotName(HitPotState state); 
