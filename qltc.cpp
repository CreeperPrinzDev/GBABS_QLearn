#include "qltc.h"
#include <fstream>

static uint32_t readU32LE(const uint8_t* p) {
    return uint32_t(p[0]) |
        (uint32_t(p[1]) << 8) |
        (uint32_t(p[2]) << 16) |
        (uint32_t(p[3]) << 24);
}

static uint16_t readU16LE(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

static void writeU32LE(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
    out.push_back(uint8_t((v >> 16) & 0xFF));
    out.push_back(uint8_t((v >> 24) & 0xFF));
}

static void writeU16LE(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t((v >> 8) & 0xFF));
}

bool readQltc(const std::string& path, QltcFile& out, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "Datei konnte nicht geöffnet werden.";
        return false;
    }

    std::vector<uint8_t> data(
        (std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>()
    );

    if (data.size() < 12) {
        error = "Datei zu klein für QLTC.";
        return false;
    }

    if (data[0] != 'Q' || data[1] != 'L' || data[2] != 'T' || data[3] != 'C') {
        error = "Magic ist nicht QLTC.";
        return false;
    }

    out.length = readU32LE(&data[4]);
    out.sampleRate = readU16LE(&data[8]);
    out.originId = data[10];

    if (data[11] != 0xFF) {
        error = "HeaderEnd ist nicht 0xFF.";
        return false;
    }

    const size_t payloadOffset = 12;
    const size_t actualPayloadSize = data.size() - payloadOffset;

    if (actualPayloadSize != out.length) {
        error = "Payload-Länge passt nicht zum Length-Feld.";
        return false;
    }

    out.payload.assign(data.begin() + payloadOffset, data.end());
    return validateQltc(out, error);
}

bool writeQltc(const std::string& path, const QltcFile& file, std::string& error) {
    if (file.originId != 'R' && file.originId != 'T') {
        error = "OriginID muss 'R' oder 'T' sein.";
        return false;
    }

    if (file.payload.size() > UINT32_MAX) {
        error = "Payload zu groß.";
        return false;
    }

    std::vector<uint8_t> data;
    data.reserve(12 + file.payload.size());

    data.push_back('Q');
    data.push_back('L');
    data.push_back('T');
    data.push_back('C');

    writeU32LE(data, static_cast<uint32_t>(file.payload.size()));
    writeU16LE(data, file.sampleRate);

    data.push_back(file.originId);
    data.push_back(0xFF);

    data.insert(data.end(), file.payload.begin(), file.payload.end());

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        error = "Datei konnte nicht geschrieben werden.";
        return false;
    }

    f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

bool validateQltc(const QltcFile& file, std::string& error) {
    if (file.originId != 'R' && file.originId != 'T') {
        error = "Ungültige OriginID.";
        return false;
    }

    if (file.length != file.payload.size()) {
        error = "Length stimmt nicht mit Payload-Größe überein.";
        return false;
    }

    if (file.sampleRate == 0) {
        error = "Samplerate darf nicht 0 sein.";
        return false;
    }

    return true;
}