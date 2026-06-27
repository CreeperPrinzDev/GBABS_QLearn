#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

inline uint8_t bucketF32_4bit(float x) {
    x = std::clamp(x, -1.0f, 1.0f);

    const float normalized = (x + 1.0f) * 0.5f;
    int bucket = static_cast<int>(normalized * 16.0f);

    if (bucket < 0) {
        bucket = 0;
    }

    if (bucket > 15) {
        bucket = 15;
    }

    return static_cast<uint8_t>(bucket);
}

inline uint8_t bucket01_4bit(float x) {
    x = std::clamp(x, 0.0f, 1.0f);

    int bucket = static_cast<int>(x * 16.0f);

    if (bucket < 0) {
        bucket = 0;
    }

    if (bucket > 15) {
        bucket = 15;
    }

    return static_cast<uint8_t>(bucket);
}

inline float pcmEnergy01(float l, float r) {
    return std::clamp(std::max(std::abs(l), std::abs(r)), 0.0f, 1.0f);
}

inline float pcmSide(float l, float r) {
    return std::clamp((l - r) * 0.5f, -1.0f, 1.0f);
}

inline uint32_t makePcmPState24(
    float currL,
    float currR,
    float prevL,
    float prevR,
    uint8_t prevByte
) {
    const uint8_t cL = bucketF32_4bit(currL);
    const uint8_t cR = bucketF32_4bit(currR);
    const uint8_t pL = bucketF32_4bit(prevL);
    const uint8_t pR = bucketF32_4bit(prevR);

    return
        (static_cast<uint32_t>(prevByte) << 16) |
        (static_cast<uint32_t>(cL) << 12) |
        (static_cast<uint32_t>(cR) << 8) |
        (static_cast<uint32_t>(pL) << 4) |
        static_cast<uint32_t>(pR);
}

inline uint32_t makePcmEnergySideState32(
    float currL,
    float currR,
    float prevL,
    float prevR,
    uint8_t prevByte
) {
    const uint8_t cL = bucketF32_4bit(currL);
    const uint8_t cR = bucketF32_4bit(currR);

    const uint8_t pL = bucketF32_4bit(prevL);
    const uint8_t pR = bucketF32_4bit(prevR);

    const uint8_t energy = bucket01_4bit(pcmEnergy01(currL, currR));
    const uint8_t curSide = bucketF32_4bit(pcmSide(currL, currR));

    return
        (static_cast<uint32_t>(prevByte) << 24) |
        (static_cast<uint32_t>(cL) << 20) |
        (static_cast<uint32_t>(cR) << 16) |
        (static_cast<uint32_t>(pL) << 12) |
        (static_cast<uint32_t>(pR) << 8) |
        (static_cast<uint32_t>(energy) << 4) |
        static_cast<uint32_t>(curSide);
}