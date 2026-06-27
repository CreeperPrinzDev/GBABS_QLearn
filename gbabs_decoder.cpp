#include "gbabs_decoder.h"
#include "wav_writer.h"

#include <algorithm>
#include <cmath>

double GbabsDecoder::calculateCapRate(uint16_t sampleRate) {
    return std::pow(0.999958, 4194304.0 / static_cast<double>(sampleRate));
}

double GbabsDecoder::highPass(double in, double& capacitor, double capRate) {
    const double out = in - capacitor;
    capacitor = in - out * capRate;
    return out;
}

std::vector<int16_t> GbabsDecoder::decodePayloadToStereo16(
    const std::vector<uint8_t>& payload,
    uint16_t sampleRate
) {
    std::vector<int16_t> pcm;
    pcm.reserve(payload.size() * 2);

    double leftCap = 0.0;
    double rightCap = 0.0;
    const double capRate = calculateCapRate(sampleRate);

    for (size_t i = 0; i < payload.size(); ++i) {
        const uint8_t sample = payload[i];

        const bool n = (sample & 0x80) != 0;       // Aktiv-/Normal-Bit
        const uint8_t l = (sample >> 4) & 0x07;    // Left
        const bool s = (sample & 0x08) != 0;       // Sign
        const uint8_t r = sample & 0x07;           // Right

        double left = 0.0;
        double right = 0.0;

        if (n) {
            left = std::round((static_cast<double>(l) + 1.0) * 127.0 / 8.0) * (s ? -1.0 : 1.0);
            right = std::round((static_cast<double>(r) + 1.0) * 127.0 / 8.0) * (s ? -1.0 : 1.0);
        }
        else {
            left = std::round((3.0 - static_cast<double>(l)) * 127.0 / 60.0);
            right = std::round((3.0 - static_cast<double>(r)) * 127.0 / 60.0);
        }

        left = highPass(left, leftCap, capRate);
        right = highPass(right, rightCap, capRate);

        int leftOut = static_cast<int>(std::round(left));
        int rightOut = static_cast<int>(std::round(right));

        leftOut = std::clamp(leftOut, -128, 127);
        rightOut = std::clamp(rightOut, -128, 127);

        // signed 8-bit-artiger Wertebereich auf 16-bit PCM skalieren
        const int16_t left16 = static_cast<int16_t>(std::clamp(leftOut * 256, -32768, 32767));
        const int16_t right16 = static_cast<int16_t>(std::clamp(rightOut * 256, -32768, 32767));

        pcm.push_back(left16);
        pcm.push_back(right16);
    }

    return pcm;
}

GbabsDecoder::Result GbabsDecoder::decodeQltcToWav(
    const std::string& inputQltcPath,
    const std::string& outputWavPath
) {
    Result result;

    QltcFile qltc;
    std::string error;

    if (!readQltc(inputQltcPath, qltc, error)) {
        result.error = error;
        return result;
    }

    auto pcm = decodePayloadToStereo16(qltc.payload, qltc.sampleRate);

    if (!writeStereo16Wav(outputWavPath, qltc.sampleRate, pcm, error)) {
        result.error = error;
        return result;
    }

    result.ok = true;
    result.frames = static_cast<uint32_t>(qltc.payload.size());
    return result;
}