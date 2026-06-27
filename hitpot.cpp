#include "hitpot.h"
#include "gbabs_symbols.h"
#include <cstdlib>

static int rewardFromState(HitPotState s) {
    switch (s) {
    case HitPotState::POT_HIT:     return 100;
    case HitPotState::VERY_HOT:    return 85;
    case HitPotState::HOT:         return 70;
    case HitPotState::MEDIUM:      return 45;
    case HitPotState::COLD:        return 20;
    case HitPotState::BLIZZARD:    return 5;
    case HitPotState::ZERO_KELVIN: return 0;
    default:                       return 0;
    }
}

HitPotResult scoreSymbol(uint8_t teacher, uint8_t student) {
    if (isForbiddenGbabsByte(student)) {
        return { HitPotState::ZERO_KELVIN, 0, 999 };
    }

    if (teacher == student) {
        return { HitPotState::POT_HIT, 100, 0 };
    }

    int d = std::abs(int(teacher) - int(student));

    // v0: numerische Näherung.
    // Später ersetzen wir das durch echte GBABS-symbolsemantische Distanz.
    HitPotState state;

    if (d <= 1) state = HitPotState::VERY_HOT;
    else if (d <= 3) state = HitPotState::HOT;
    else if (d <= 8) state = HitPotState::MEDIUM;
    else if (d <= 20) state = HitPotState::COLD;
    else if (d <= 48) state = HitPotState::BLIZZARD;
    else state = HitPotState::ZERO_KELVIN;

    return { state, rewardFromState(state), d };
}

std::string hitPotName(HitPotState state) {
    switch (state) {
    case HitPotState::POT_HIT:     return "POT_HIT";
    case HitPotState::VERY_HOT:    return "VERY_HOT";
    case HitPotState::HOT:         return "HOT";
    case HitPotState::MEDIUM:      return "MEDIUM";
    case HitPotState::COLD:        return "COLD";
    case HitPotState::BLIZZARD:    return "BLIZZARD";
    case HitPotState::ZERO_KELVIN: return "ZERO_KELVIN";
    default:                       return "UNKNOWN";
    }
}