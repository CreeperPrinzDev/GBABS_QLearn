#include "teacher_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

uint8_t TeacherEncoder::q3(float x) {
    x = std::clamp(x, 0.0f, 1.0f);

    int v = static_cast<int>(std::round(x * 7.0f));
    v = std::clamp(v, 0, 7);

    return static_cast<uint8_t>(v);
}

float TeacherEncoder::softclip(float x) {
    constexpr float k = 1.2f;
    const float y = std::tanh(k * x) / std::tanh(k);
    return std::clamp(y, -1.0f, 1.0f);
}

bool TeacherEncoder::runFfmpegPreprocess(
    const std::string& inputAudioPath,
    const std::string& pcmPath,
    uint16_t sampleRate,
    std::string& error
) {
    std::stringstream ffm;

    ffm << "ffmpeg -y -i \"" << inputAudioPath
        << "\" -ac 2 "
        << "-af \"highpass=f=40,"
        << "lowpass=f=7500,"
        << "equalizer=f=100:t=q:w=1:g=-6,"
        << "dynaudnorm=f=75:p=0.75:g=5:m=15,"
        << "alimiter=limit=0.97,"
        << "aresample=" << sampleRate
        << ":resampler=soxr:precision=33:dither_method=none,"
        << "afftdn=nf=-28\" "
        << "-f f32le -acodec pcm_f32le -ar " << sampleRate
        << " \"" << pcmPath << "\"";

    const int rc = std::system(ffm.str().c_str());

    if (rc != 0) {
        error = "ffmpeg preprocessing failed.";
        return false;
    }

    return true;
}

bool TeacherEncoder::loadF32StereoPcm(
    const std::string& pcmPath,
    std::vector<float>& interleaved,
    uint32_t& frames,
    std::string& error
) {
    std::ifstream src(pcmPath, std::ios::binary);

    if (!src) {
        error = "Temp PCM not found.";
        return false;
    }

    src.seekg(0, std::ios::end);
    const std::streamoff bytes = src.tellg();
    src.seekg(0, std::ios::beg);

    if (bytes <= 0 || (bytes % (2 * sizeof(float))) != 0) {
        error = "PCM size invalid. Expected f32le stereo.";
        return false;
    }

    const uint64_t frameCount64 = static_cast<uint64_t>(bytes) / (2ULL * sizeof(float));

    if (frameCount64 > UINT32_MAX) {
        error = "PCM too large for u32 frame counter.";
        return false;
    }

    frames = static_cast<uint32_t>(frameCount64);
    interleaved.resize(static_cast<size_t>(frames) * 2);

    src.read(
        reinterpret_cast<char*>(interleaved.data()),
        bytes
    );

    if (!src) {
        error = "Failed to read PCM data.";
        return false;
    }

    return true;
}

void TeacherEncoder::repairGlobalPhase(std::vector<float>& interleaved, uint32_t frames) {
    long double acc = 0.0;

    for (uint32_t i = 0; i < frames; ++i) {
        acc += static_cast<long double>(interleaved[2ULL * i])
            * static_cast<long double>(interleaved[2ULL * i + 1]);
    }

    if (acc < 0.0) {
        for (uint32_t i = 0; i < frames; ++i) {
            interleaved[2ULL * i + 1] = -interleaved[2ULL * i + 1];
        }
    }
}

std::vector<uint8_t> TeacherEncoder::encodeIrbmsPayload(
    const std::vector<float>& interleaved,
    uint32_t frames,
    float gate
) {
    std::vector<uint8_t> payload;
    payload.resize(frames);

    for (uint32_t i = 0; i < frames; ++i) {
        const float l = interleaved[2ULL * i];
        const float r = interleaved[2ULL * i + 1];

        float core = 0.5f * (l + r);
        core = softclip(core);
        core = std::clamp(core, -0.84f, 0.84f);

        const float absL = std::fabs(l);
        const float absR = std::fabs(r);
        const float energy = std::max(absL, absR);

        const float denom = std::max(std::max(absL, absR), 1e-12f);

        const bool sign = core < 0.0f;
        const float mag = std::fabs(core);

        const uint8_t lvll = q3(mag * (absL / denom));
        const uint8_t lvlr = q3(mag * (absR / denom));

        if (energy < gate) {
            auto approxLevel = [&](float sample) -> uint8_t {
                int q = static_cast<int>(std::floor(std::fabs(sample) * 15.0f));
                q = std::clamp(q, 0, 4);

                int approx = 3 + (sign ? q : -q);

                return static_cast<uint8_t>(
                    std::clamp(std::abs(approx), 0, 7)
                    );
                };

            const uint8_t nl = approxLevel(l);
            const uint8_t nr = approxLevel(r);

            uint8_t b = 0;
            b |= static_cast<uint8_t>((nl & 0x07) << 4);
            b |= static_cast<uint8_t>(nr & 0x07);

            payload[i] = b;
            continue;
        }

        uint8_t b = 0;
        b |= 0x80;
        b |= static_cast<uint8_t>((lvll & 0x07) << 4);

        if (sign) {
            b |= 0x08;
        }

        b |= static_cast<uint8_t>(lvlr & 0x07);

        payload[i] = b;
    }

    return payload;
}

TeacherEncoder::Result TeacherEncoder::encodeToQltc(
    const std::string& inputAudioPath,
    const std::string& outputQltcPath,
    const Options& options
) {
    Result result;

    std::string error;

    if (!runFfmpegPreprocess(inputAudioPath, options.tempPcmPath, options.sampleRate, error)) {
        result.error = error;
        return result;
    }

    std::vector<float> pcm;
    uint32_t frames = 0;

    if (!loadF32StereoPcm(options.tempPcmPath, pcm, frames, error)) {
        result.error = error;
        return result;
    }

    repairGlobalPhase(pcm, frames);

    QltcFile qltc;
    qltc.sampleRate = options.sampleRate;
    qltc.originId = 'R';
    qltc.payload = encodeIrbmsPayload(pcm, frames, options.lesaGate);
    qltc.length = static_cast<uint32_t>(qltc.payload.size());

    if (!writeQltc(outputQltcPath, qltc, error)) {
        result.error = error;
        return result;
    }

    if (!options.keepTempPcm) {
        std::remove(options.tempPcmPath.c_str());
    }

    result.ok = true;
    result.frames = frames;
    result.payloadBytes = static_cast<uint32_t>(qltc.payload.size());

    return result;
}