#include "qtable.h"
#include "gbabs_symbols.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>

QTable::QTable() {
    buildActionMap();
    reset();
}

void QTable::buildActionMap() {
    byteToActionIndex.fill(-1);

    size_t idx = 0;

    for (int b = 0; b < 256; ++b) {
        const uint8_t byte = static_cast<uint8_t>(b);

        if (!isValidGbabsByte(byte)) {
            continue;
        }

        if (idx < ACTION_COUNT) {
            actions[idx] = byte;
            byteToActionIndex[byte] = static_cast<int16_t>(idx);
            idx++;
        }
    }
}

void QTable::reset() {
    q.clear();
}

size_t QTable::actionCount() const {
    return ACTION_COUNT;
}

size_t QTable::stateCount() const {
    return q.size();
}

uint8_t QTable::actionToByte(size_t actionIndex) const {
    return actions[actionIndex];
}

size_t QTable::byteToAction(uint8_t byte) const {
    const int16_t idx = byteToActionIndex[byte];

    if (idx < 0) {
        return SIZE_MAX;
    }

    return static_cast<size_t>(idx);
}

QTable::Row& QTable::getOrCreateRow(uint32_t state) {
    auto it = q.find(state);

    if (it != q.end()) {
        return it->second;
    }

    Row row;
    row.q.fill(0.0f);
    row.bestValue = 0.0f;
    row.bestAction = 0;

    auto inserted = q.emplace(state, std::move(row));
    return inserted.first->second;
}

const QTable::Row* QTable::getRow(uint32_t state) const {
    auto it = q.find(state);

    if (it == q.end()) {
        return nullptr;
    }

    return &it->second;
}

void QTable::refreshBest(Row& row) {
    size_t best = 0;
    float bestValue = row.q[0];

    for (size_t a = 1; a < ACTION_COUNT; ++a) {
        if (row.q[a] > bestValue) {
            bestValue = row.q[a];
            best = a;
        }
    }

    row.bestValue = bestValue;
    row.bestAction = best;
}

float QTable::get(uint32_t state, size_t action) const {
    const Row* row = getRow(state);

    if (!row) {
        return 0.0f;
    }

    return row->q[action];
}

void QTable::update(
    uint32_t state,
    size_t action,
    float reward,
    uint32_t nextState,
    float alpha,
    float gamma
) {
    Row& row = getOrCreateRow(state);

    const float oldValue = row.q[action];

    const Row* nextRow = getRow(nextState);
    const float bestNext = nextRow ? nextRow->bestValue : 0.0f;

    const float target = reward + gamma * bestNext;

    const float newValue =
        oldValue + alpha * (target - oldValue);

    row.q[action] = newValue;

    if (action == row.bestAction) {
        if (newValue >= row.bestValue) {
            row.bestValue = newValue;
        }
        else {
            refreshBest(row);
        }
    }
    else if (newValue > row.bestValue) {
        row.bestValue = newValue;
        row.bestAction = action;
    }
}

size_t QTable::bestAction(uint32_t state) const {
    const Row* row = getRow(state);

    if (!row) {
        return 0;
    }

    return row->bestAction;
}

bool QTable::save(const std::string& path, std::string& error) const {
    std::ofstream f(path, std::ios::binary);

    if (!f) {
        error = "Open failed.";
        return false;
    }

    f.write("QTBL", 4);

    const uint64_t count = static_cast<uint64_t>(q.size());
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [state, row] : q) {
        f.write(reinterpret_cast<const char*>(&state), sizeof(state));

        f.write(
            reinterpret_cast<const char*>(row.q.data()),
            sizeof(float) * ACTION_COUNT
        );

        f.write(
            reinterpret_cast<const char*>(&row.bestValue),
            sizeof(row.bestValue)
        );

        const uint64_t bestAction64 = static_cast<uint64_t>(row.bestAction);
        f.write(reinterpret_cast<const char*>(&bestAction64), sizeof(bestAction64));
    }

    return true;
}

bool QTable::load(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);

    if (!f) {
        error = "Open failed.";
        return false;
    }

    char magic[4]{};
    f.read(magic, 4);

    if (std::memcmp(magic, "QTBL", 4) != 0) {
        error = "Bad magic.";
        return false;
    }

    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), sizeof(count));

    q.clear();

    for (uint64_t i = 0; i < count; ++i) {
        uint32_t state = 0;
        Row row;
        row.q.fill(0.0f);

        f.read(reinterpret_cast<char*>(&state), sizeof(state));

        f.read(
            reinterpret_cast<char*>(row.q.data()),
            sizeof(float) * ACTION_COUNT
        );

        f.read(
            reinterpret_cast<char*>(&row.bestValue),
            sizeof(row.bestValue)
        );

        uint64_t bestAction64 = 0;
        f.read(reinterpret_cast<char*>(&bestAction64), sizeof(bestAction64));

        row.bestAction = static_cast<size_t>(bestAction64);

        if (!f) {
            error = "Read failed.";
            return false;
        }

        if (row.bestAction >= ACTION_COUNT) {
            refreshBest(row);
        }

        q.emplace(state, std::move(row));
    }

    return true;
}