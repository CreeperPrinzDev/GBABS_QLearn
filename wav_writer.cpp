#include "wav_writer.h"

#include <fstream>

static void writeU16LE(std::ofstream& f, uint16_t v) {
    f.put(static_cast<char>(v & 0xFF));
    f.put(static_cast<char>((v >> 8) & 0xFF));
}

static void writeU32LE(std::ofstream& f, uint32_t v) {
    f.put(static_cast<char>(v & 0xFF));
    f.put(static_cast<char>((v >> 8) & 0xFF));
    f.put(static_cast<char>((v >> 16) & 0xFF));
    f.put(static_cast<char>((v >> 24) & 0xFF));
}

bool writeStereo16Wav(
    const std::string& path,
    uint16_t sampleRate,
    const std::vector<int16_t>& interleavedStereo,
    std::string& error
) {
    if ((interleavedStereo.size() % 2) != 0) {
        error = "Stereo buffer has odd sample count.";
        return false;
    }

    const uint16_t channels = 2;
    const uint16_t bitsPerSample = 16;
    const uint16_t blockAlign = channels * bitsPerSample / 8;
    const uint32_t byteRate = static_cast<uint32_t>(sampleRate) * blockAlign;
    const uint32_t dataSize = static_cast<uint32_t>(interleavedStereo.size() * sizeof(int16_t));
    const uint32_t riffSize = 36 + dataSize;

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "Could not open WAV output.";
        return false;
    }

    f.write("RIFF", 4);
    writeU32LE(f, riffSize);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    writeU32LE(f, 16);                  // PCM fmt chunk size
    writeU16LE(f, 1);                   // PCM format
    writeU16LE(f, channels);
    writeU32LE(f, sampleRate);
    writeU32LE(f, byteRate);
    writeU16LE(f, blockAlign);
    writeU16LE(f, bitsPerSample);

    f.write("data", 4);
    writeU32LE(f, dataSize);

    f.write(
        reinterpret_cast<const char*>(interleavedStereo.data()),
        static_cast<std::streamsize>(dataSize)
    );

    return true;
}