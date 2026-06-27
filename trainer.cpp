#include "trainer.h"
#include "gbabs_symbols.h"
#include "qltc.h"
#include "hitpot.h"
#include "qtable.h"
#include "pcm_state.h"

#include <array>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <random>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <cmath>

namespace {
    constexpr size_t TOPF_POT_HIT = 0;
    constexpr size_t TOPF_VERY_HOT = 1;
    constexpr size_t TOPF_HOT = 2;
    constexpr size_t TOPF_MEDIUM = 3;
    constexpr size_t TOPF_COLD = 4;
    constexpr size_t TOPF_BLIZZARD = 5;
    constexpr size_t TOPF_ZERO_KELVIN = 6;

    const char* topfName(size_t index) {
        switch (index) {
        case TOPF_POT_HIT: return "POT_HIT";
        case TOPF_VERY_HOT: return "VERY_HOT";
        case TOPF_HOT: return "HOT";
        case TOPF_MEDIUM: return "MEDIUM";
        case TOPF_COLD: return "COLD";
        case TOPF_BLIZZARD: return "BLIZZARD";
        case TOPF_ZERO_KELVIN: return "ZERO_KELVIN";
        default: return "UNKNOWN";
        }
    }

    size_t topfIndexFromReward(const HitPotResult& r) {
        switch (r.reward) {
        case 100: return TOPF_POT_HIT;
        case 85: return TOPF_VERY_HOT;
        case 70: return TOPF_HOT;
        case 45: return TOPF_MEDIUM;
        case 20: return TOPF_COLD;
        case 5: return TOPF_BLIZZARD;
        case 0:
        default:
            return TOPF_ZERO_KELVIN;
        }
    }

    uint64_t topfTotal(const std::array<uint64_t, 7>& counts) {
        uint64_t total = 0;

        for (uint64_t v : counts) {
            total += v;
        }

        return total;
    }

    uint64_t topfAccuracyHits(const std::array<uint64_t, 7>& counts) {
        return counts[TOPF_POT_HIT]
            + counts[TOPF_VERY_HOT]
            + counts[TOPF_HOT];
    }

    double topfPercent(uint64_t value, uint64_t total) {
        if (total == 0) {
            return 0.0;
        }

        return static_cast<double>(value) * 100.0 / static_cast<double>(total);
    }

    double topfAccuracy(const std::array<uint64_t, 7>& counts) {
        const uint64_t total = topfTotal(counts);

        if (total == 0) {
            return 0.0;
        }

        return static_cast<double>(topfAccuracyHits(counts)) * 100.0
            / static_cast<double>(total);
    }

    void printEpisodeTopfStats(
        uint32_t episode,
        uint32_t episodes,
        bool studentForcing,
        const std::array<uint64_t, 7>& episodeCounts,
        const std::array<uint64_t, 7>& runCounts
    ) {
        const uint64_t episodeTotal = topfTotal(episodeCounts);

        std::cout << "[EPSTATS] Episode "
            << (episode + 1)
            << "/"
            << episodes
            << " mode="
            << (studentForcing ? "student" : "teacher")
            << " samples="
            << episodeTotal
            << " Acc="
            << std::fixed
            << std::setprecision(4)
            << topfAccuracy(episodeCounts)
            << "%"
            << " RunAcc="
            << topfAccuracy(runCounts)
            << "%"
            << "\n";

        std::cout << "[EPSTATS] ";

        for (size_t i = 0; i < episodeCounts.size(); ++i) {
            if (i > 0) {
                std::cout << " ";
            }

            std::cout << topfName(i)
                << "="
                << episodeCounts[i]
                << " ("
                << std::fixed
                << std::setprecision(2)
                << topfPercent(episodeCounts[i], episodeTotal)
                << "%)";
        }

        std::cout << "\n";
    }

    void printTopfLine(
        const char* prefix,
        const std::array<uint64_t, 7>& counts
    ) {
        const uint64_t total = topfTotal(counts);

        std::cout << prefix;

        for (size_t i = 0; i < counts.size(); ++i) {
            if (i > 0) {
                std::cout << " ";
            }

            std::cout << topfName(i)
                << "="
                << counts[i]
                << " ("
                << std::fixed
                << std::setprecision(2)
                << topfPercent(counts[i], total)
                << "%)";
        }

        std::cout << "\n";
    }

}

Trainer::Result Trainer::trainRandomBaseline(
    const std::string& datasetFolder,
    const Options& options
) {
    Result result;

    validSymbols = buildValidSymbols();
    rng.seed(options.seed);

    if (validSymbols.size() != 192) {
        result.error = "Valid symbol count is not 192. GBABS mask broken.";
        return result;
    }

    std::vector<std::vector<uint8_t>> payloads;
    std::string error;

    if (!loadDatasetPayloads(datasetFolder, payloads, error)) {
        result.error = error;
        return result;
    }

    result.filesLoaded = payloads.size();

    for (uint32_t ep = 0; ep < options.episodes; ++ep) {
        std::cout << "[TRAIN] Episode " << (ep + 1) << "/" << options.episodes << "\n";

        for (const auto& teacherPayload : payloads) {
            for (uint8_t teacherByte : teacherPayload) {
                const uint8_t studentByte = randomValidByte();
                const HitPotResult r = scoreSymbol(teacherByte, studentByte);
                result.stats.add(r);
            }
        }
    }

    result.ok = true;
    return result;
}

Trainer::Result Trainer::trainQTable(
    const std::string& datasetFolder,
    const Options& options
) {
    Result result;

    validSymbols = buildValidSymbols();
    rng.seed(options.seed);

    if (validSymbols.size() != 192) {
        result.error = "Valid symbol count is not 192. GBABS mask broken.";
        return result;
    }

    std::vector<DatasetEntry> entries;
    std::string error;

    if (!scanTrainingDataset(datasetFolder, entries, error)) {
        result.error = error;
        return result;
    }

    result.filesLoaded = entries.size();

    const uint32_t requestedWorkingSet =
        options.workingSetSize == 0 ? 1 : options.workingSetSize;

    const uint32_t workingSetSize = static_cast<uint32_t>(
        std::min<size_t>(requestedWorkingSet, entries.size())
        );

    const auto schedule = buildBalancedSchedule(
        entries.size(),
        workingSetSize,
        options.episodes
    );

    const uint32_t studentSwitchEpisode = options.reset
        ? computeStudentSwitchEpisode(
            entries.size(),
            workingSetSize,
            options.episodes,
            options.teacherForcingRatio
        )
        : 0;

    const bool studentSwitchEnabled =
        options.useStudentForcing && studentSwitchEpisode < options.episodes;

    auto printSlotStats = [&](const char* label, const std::vector<uint64_t>& counts) {
        if (counts.empty()) {
            return;
        }

        const auto minmax = std::minmax_element(counts.begin(), counts.end());
        const uint64_t minValue = *minmax.first;
        const uint64_t maxValue = *minmax.second;

        uint64_t sum = 0;

        for (uint64_t v : counts) {
            sum += v;
        }

        const double avg =
            static_cast<double>(sum) / static_cast<double>(counts.size());

        std::cout << "[STATS] " << label
            << " total=" << sum
            << " perPairAvg=" << std::fixed << std::setprecision(2) << avg
            << " perPairMin=" << minValue
            << " perPairMax=" << maxValue;

        if (minValue == maxValue) {
            std::cout << " exact";
        }
        else {
            std::cout << " spread=" << (maxValue - minValue);
        }

        std::cout << "\n";
        };

    auto printBetterTrainingStats = [&]() {
        const size_t fileCount = entries.size();

        if (fileCount == 0 || workingSetSize == 0 || options.episodes == 0) {
            return;
        }

        std::vector<uint64_t> totalPerPair(fileCount, 0);
        std::vector<uint64_t> teacherPerPair(fileCount, 0);
        std::vector<uint64_t> studentPerPair(fileCount, 0);

        for (uint32_t ep = 0; ep < options.episodes; ++ep) {
            const bool studentForcing =
                studentSwitchEnabled && ep >= studentSwitchEpisode;

            for (size_t fileIndex : schedule[ep]) {
                totalPerPair[fileIndex]++;

                if (studentForcing) {
                    studentPerPair[fileIndex]++;
                }
                else {
                    teacherPerPair[fileIndex]++;
                }
            }
        }

        const uint64_t totalSlots =
            static_cast<uint64_t>(workingSetSize) * static_cast<uint64_t>(options.episodes);

        const uint32_t gcdValue = std::gcd(
            static_cast<uint32_t>(fileCount),
            workingSetSize
        );

        const uint32_t fairPeriod = gcdValue == 0
            ? 0
            : static_cast<uint32_t>(fileCount) / gcdValue;

        const double passesPerPair =
            static_cast<double>(totalSlots) / static_cast<double>(fileCount);

        std::cout << "[STATS] Better Training Stats\n";

        std::cout << "[STATS] Pairs=" << fileCount
            << " WorkingSet=" << workingSetSize
            << " Episodes=" << options.episodes
            << " Slots=" << totalSlots
            << "\n";

        std::cout << "[STATS] FairPeriod=" << fairPeriod
            << " episodes"
            << " FullDatasetPasses=" << std::fixed << std::setprecision(2) << passesPerPair
            << "\n";

        if (studentSwitchEnabled) {
            const uint32_t teacherEpisodes = studentSwitchEpisode;
            const uint32_t studentEpisodes = options.episodes - studentSwitchEpisode;

            std::cout << "[STATS] TeacherEpisodes=" << teacherEpisodes
                << " StudentEpisodes=" << studentEpisodes
                << "\n";

            std::cout << "[STATS] TeacherRatioActual="
                << std::fixed << std::setprecision(4)
                << (static_cast<double>(teacherEpisodes) / static_cast<double>(options.episodes) * 100.0)
                << "% StudentRatioActual="
                << (static_cast<double>(studentEpisodes) / static_cast<double>(options.episodes) * 100.0)
                << "%\n";
        }
        else {
            std::cout << "[STATS] TeacherEpisodes=" << options.episodes
                << " StudentEpisodes=0\n";
        }

        printSlotStats("AllPasses", totalPerPair);
        printSlotStats("TeacherPasses", teacherPerPair);
        printSlotStats("StudentPasses", studentPerPair);
        };

    QTable table;

    if (!options.reset) {
        if (!table.load(options.modelPath, error)) {
            result.error = "Could not resume QTable: " + error;
            return result;
        }
    }

    float epsilon = options.reset
        ? options.epsilon
        : 0.20f;

    std::vector<std::filesystem::path> createdPcmCaches;

    std::cout << "[SCHED] Files=" << entries.size()
        << " WorkingSet=" << workingSetSize
        << " Episodes=" << options.episodes
        << " Slots=" << (static_cast<uint64_t>(workingSetSize) * options.episodes)
        << "\n";

    printBetterTrainingStats();

    if (studentSwitchEnabled) {
        const uint32_t teacherEpisodes = studentSwitchEpisode;
        const uint32_t studentEpisodes = options.episodes - studentSwitchEpisode;

        if (!options.reset && studentSwitchEpisode == 0) {
            std::cout << "[MODE] Resume detected. Using STUDENT forcing from episode 1.\n";
        }
        else if (teacherEpisodes > 0) {
            std::cout << "[MODE] Teacher forcing episodes: 1-" << teacherEpisodes << "\n";
            std::cout << "[MODE] Student forcing episodes: " << (studentSwitchEpisode + 1)
                << "-" << options.episodes << "\n";
        }
        else {
            std::cout << "[MODE] Student forcing episodes: 1-" << options.episodes << "\n";
        }

        std::cout << "[MODE] Student epsilon boost: "
            << (options.boostEpsilonOnStudentSwitch ? options.studentForcingEpsilon : 0.0f)
            << "\n";

        if (workingSetSize > 0) {
            const uint64_t teacherSlots =
                static_cast<uint64_t>(teacherEpisodes) * workingSetSize;
            const uint64_t studentSlots =
                static_cast<uint64_t>(studentEpisodes) * workingSetSize;

            std::cout << "[MODE] Teacher slots=" << teacherSlots
                << " Student slots=" << studentSlots
                << "\n";
        }
    }
    else {
        std::cout << "[MODE] Student forcing disabled or no valid switch point. Using teacher forcing only.\n";
    }

    std::array<uint64_t, 7> runTopfCounts{};

    for (uint32_t ep = 0; ep < options.episodes; ++ep) {
        const bool studentForcing =
            studentSwitchEnabled && ep >= studentSwitchEpisode;

        std::array<uint64_t, 7> episodeTopfCounts{};

        if (studentSwitchEnabled && ep == studentSwitchEpisode) {
            if (options.boostEpsilonOnStudentSwitch) {
                epsilon = std::max(epsilon, options.studentForcingEpsilon);
            }

            std::cout << "[MODE] Switching to STUDENT forcing at episode "
                << (ep + 1)
                << " epsilon=" << epsilon
                << "\n";
        }

        const auto& group = schedule[ep];

        std::cout << "[TRAIN] Episode " << (ep + 1) << "/" << options.episodes
            << " epsilon=" << epsilon
            << " files=" << group.size()
            << " mode=" << (studentForcing ? "student" : "teacher")
            << "\n";

        std::vector<TrainingItem> activeItems;
        activeItems.reserve(group.size());

        for (const size_t fileIndex : group) {
            TrainingItem item;
            std::string loadError;

            if (!loadTrainingItem(entries[fileIndex], options, item, createdPcmCaches, loadError)) {
                std::cout << "[WARN] Skipping "
                    << entries[fileIndex].qltcPath.filename().string()
                    << ": " << loadError << "\n";
                continue;
            }

            std::cout << "[DATA] Active " << item.name
                << " frames=" << item.frames
                << "\n";

            activeItems.push_back(std::move(item));
        }

        std::shuffle(activeItems.begin(), activeItems.end(), rng);

        for (const auto& item : activeItems) {
            trainItem(
                item,
                table,
                epsilon,
                options,
                studentForcing,
                result.stats,
                &episodeTopfCounts,
                &runTopfCounts
            );
        }

        printEpisodeTopfStats(
            ep,
            options.episodes,
            studentForcing,
            episodeTopfCounts,
            runTopfCounts
        );

        activeItems.clear();
        activeItems.shrink_to_fit();

        epsilon = std::max(options.epsilonMin, epsilon * options.epsilonDecay);
    }

    if (!table.save(options.modelPath, error)) {
        result.error = "Could not save QTable: " + error;
        return result;
    }

    if (options.usePcmCache && !options.keepPcmCache) {
        std::sort(createdPcmCaches.begin(), createdPcmCaches.end());
        createdPcmCaches.erase(
            std::unique(createdPcmCaches.begin(), createdPcmCaches.end()),
            createdPcmCaches.end()
        );

        for (const auto& path : createdPcmCaches) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    std::cout << "[RESULT] Final Training Stats\n";

    std::cout << "[RESULT] Samples="
        << topfTotal(runTopfCounts)
        << " Accuracy="
        << std::fixed
        << std::setprecision(4)
        << topfAccuracy(runTopfCounts)
        << "%\n";

    printTopfLine("[RESULT] ", runTopfCounts);

    std::cout << "Visited States: " << table.stateCount() << "\n";

    result.ok = true;
    return result;
}


void Trainer::trainItem(
    const TrainingItem& item,
    QTable& table,
    float epsilon,
    const Options& options,
    bool studentForcing,
    TrainingStats& stats,
    std::array<uint64_t, 7>* episodeTopfCounts,
    std::array<uint64_t, 7>* runTopfCounts
) {
    uint8_t prevTeacherByte = 0x00;
    uint8_t prevStudentByte = 0x00;

    float prevL = 0.0f;
    float prevR = 0.0f;

    const size_t count = std::min<size_t>(item.teacherPayload.size(), item.frames);

    for (size_t i = 0; i < count; ++i) {
        const uint8_t teacherByte = item.teacherPayload[i];

        const float currL = item.pcmInterleaved[2ULL * i];
        const float currR = item.pcmInterleaved[2ULL * i + 1];

        const uint8_t prevContextByte = studentForcing
            ? prevStudentByte
            : prevTeacherByte;

        const uint32_t state = makePcmEnergySideState32(
            currL,
            currR,
            prevL,
            prevR,
            prevContextByte
        );

        const size_t actionIndex = chooseAction(table, state, epsilon);
        const uint8_t studentByte = table.actionToByte(actionIndex);

        const HitPotResult rewardResult = scoreSymbol(teacherByte, studentByte);
        stats.add(rewardResult);

        const size_t topfIndex = topfIndexFromReward(rewardResult);

        if (episodeTopfCounts) {
            (*episodeTopfCounts)[topfIndex]++;
        }

        if (runTopfCounts) {
            (*runTopfCounts)[topfIndex]++;
        }

        uint32_t nextState = state;

        if (i + 1 < count) {
            const float nextL = item.pcmInterleaved[2ULL * (i + 1)];
            const float nextR = item.pcmInterleaved[2ULL * (i + 1) + 1];

            const uint8_t nextPrevContextByte = studentForcing
                ? studentByte
                : teacherByte;

            nextState = makePcmEnergySideState32(
                nextL,
                nextR,
                currL,
                currR,
                nextPrevContextByte
            );
        }

        table.update(
            state,
            actionIndex,
            static_cast<float>(rewardResult.reward),
            nextState,
            options.alpha,
            options.gamma
        );

        prevTeacherByte = teacherByte;
        prevStudentByte = studentByte;
        prevL = currL;
        prevR = currR;
    }
}


std::vector<std::vector<size_t>> Trainer::buildBalancedSchedule(
    size_t fileCount,
    uint32_t workingSetSize,
    uint32_t episodes
) {
    std::vector<std::vector<size_t>> schedule;
    schedule.reserve(episodes);

    if (fileCount == 0 || workingSetSize == 0 || episodes == 0) {
        return schedule;
    }

    const uint64_t totalSlots =
        static_cast<uint64_t>(workingSetSize) * static_cast<uint64_t>(episodes);

    const uint64_t base = totalSlots / static_cast<uint64_t>(fileCount);
    const uint64_t remainder = totalSlots % static_cast<uint64_t>(fileCount);

    std::vector<uint64_t> remaining(fileCount, base);

    std::vector<size_t> ids;
    ids.reserve(fileCount);

    for (size_t i = 0; i < fileCount; ++i) {
        ids.push_back(i);
    }

    std::shuffle(ids.begin(), ids.end(), rng);

    for (uint64_t i = 0; i < remainder; ++i) {
        remaining[ids[static_cast<size_t>(i)]]++;
    }

    for (uint32_t ep = 0; ep < episodes; ++ep) {
        std::vector<size_t> candidates;
        candidates.reserve(fileCount);

        for (size_t i = 0; i < fileCount; ++i) {
            if (remaining[i] > 0) {
                candidates.push_back(i);
            }
        }

        std::shuffle(candidates.begin(), candidates.end(), rng);

        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [&](size_t a, size_t b) {
                return remaining[a] > remaining[b];
            }
        );

        std::vector<size_t> group;
        group.reserve(workingSetSize);

        const size_t take = std::min<size_t>(workingSetSize, candidates.size());

        for (size_t i = 0; i < take; ++i) {
            const size_t id = candidates[i];
            group.push_back(id);
            remaining[id]--;
        }

        schedule.push_back(std::move(group));
    }

    return schedule;
}

uint32_t Trainer::computeStudentSwitchEpisode(
    size_t fileCount,
    uint32_t workingSetSize,
    uint32_t episodes,
    float teacherRatio
) const {
    if (fileCount == 0 || workingSetSize == 0 || episodes < 2) {
        return episodes;
    }

    if (teacherRatio <= 0.0f) {
        return 0;
    }

    if (teacherRatio >= 1.0f) {
        return episodes;
    }

    const uint32_t gcd = std::gcd(
        static_cast<uint32_t>(fileCount),
        workingSetSize
    );

    const uint32_t fairPeriod = static_cast<uint32_t>(fileCount) / gcd;

    if (fairPeriod == 0) {
        return episodes;
    }

    const double rawTarget = static_cast<double>(episodes) * static_cast<double>(teacherRatio);
    uint32_t target = static_cast<uint32_t>(std::floor(rawTarget + 0.5));

    if (target >= episodes) {
        target = episodes - 1;
    }

    uint32_t rounded = (target / fairPeriod) * fairPeriod;

    if (rounded == 0 && episodes > fairPeriod) {
        rounded = fairPeriod;
    }

    if (rounded >= episodes) {
        rounded = (episodes / fairPeriod) * fairPeriod;
        if (rounded >= episodes && rounded >= fairPeriod) {
            rounded -= fairPeriod;
        }
    }

    if (rounded == 0 || rounded >= episodes) {
        return episodes;
    }

    return rounded;
}

size_t Trainer::chooseAction(QTable& table, uint32_t state, float epsilon) {
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    if (coin(rng) < epsilon) {
        std::uniform_int_distribution<size_t> dist(0, table.actionCount() - 1);
        return dist(rng);
    }

    return table.bestAction(state);
}

bool Trainer::scanTrainingDataset(
    const std::string& datasetFolder,
    std::vector<DatasetEntry>& entries,
    std::string& error
) {
    namespace fs = std::filesystem;

    entries.clear();

    if (!fs::exists(datasetFolder)) {
        error = "Dataset folder does not exist.";
        return false;
    }

    if (!fs::is_directory(datasetFolder)) {
        error = "Dataset path is not a folder.";
        return false;
    }

    for (const auto& entry : fs::directory_iterator(datasetFolder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const fs::path qltcPath = entry.path();

        if (qltcPath.extension() != ".qltc") {
            continue;
        }

        fs::path wavPath = qltcPath;
        wavPath.replace_extension(".wav");

        if (!fs::exists(wavPath)) {
            std::cout << "[WARN] Skipping " << qltcPath.filename().string()
                << ": matching WAV not found.\n";
            continue;
        }

        DatasetEntry de;
        de.qltcPath = qltcPath;
        de.wavPath = wavPath;
        entries.push_back(std::move(de));

        std::cout << "[DATA] Found pair "
            << qltcPath.filename().string()
            << " + "
            << wavPath.filename().string()
            << "\n";
    }

    std::sort(
        entries.begin(),
        entries.end(),
        [](const DatasetEntry& a, const DatasetEntry& b) {
            return a.qltcPath.filename().string() < b.qltcPath.filename().string();
        }
    );

    if (entries.empty()) {
        error = "No valid .qltc/.wav pairs found in dataset folder.";
        return false;
    }

    return true;
}

std::filesystem::path Trainer::makePcmCachePath(
    const std::filesystem::path& qltcPath,
    uint16_t sampleRate
) const {
    namespace fs = std::filesystem;

    const fs::path cacheDir = qltcPath.parent_path() / ".pcmcache";

    std::ostringstream name;
    name << qltcPath.stem().string()
        << "."
        << sampleRate
        << ".stereo.f32le.pcm";

    return cacheDir / name.str();
}

bool Trainer::loadTrainingItem(
    const DatasetEntry& entry,
    const Options& options,
    TrainingItem& item,
    std::vector<std::filesystem::path>& createdPcmCaches,
    std::string& error
) {
    namespace fs = std::filesystem;

    QltcFile qltc;

    if (!readQltc(entry.qltcPath.string(), qltc, error)) {
        return false;
    }

    const uint16_t sr = qltc.sampleRate != 0
        ? qltc.sampleRate
        : options.sampleRate;

    fs::path pcmPath;
    bool createdCacheThisCall = false;

    if (options.usePcmCache) {
        pcmPath = makePcmCachePath(entry.qltcPath, sr);

        std::error_code ec;
        fs::create_directories(pcmPath.parent_path(), ec);

        if (!fs::exists(pcmPath)) {
            if (!preprocessWavToF32Stereo(entry.wavPath.string(), pcmPath.string(), sr, error)) {
                return false;
            }

            createdPcmCaches.push_back(pcmPath);
            createdCacheThisCall = true;
        }
    }
    else {
        pcmPath = entry.qltcPath.parent_path()
            / (entry.qltcPath.stem().string() + "_train_tmp_f32le.pcm");

        if (!preprocessWavToF32Stereo(entry.wavPath.string(), pcmPath.string(), sr, error)) {
            return false;
        }

        createdCacheThisCall = true;
    }

    std::vector<float> pcm;
    uint32_t frames = 0;

    if (!loadF32StereoPcm(pcmPath.string(), pcm, frames, error)) {
        if (!options.usePcmCache || createdCacheThisCall) {
            std::error_code ec;
            fs::remove(pcmPath, ec);
        }
        return false;
    }

    if (!options.usePcmCache) {
        std::error_code ec;
        fs::remove(pcmPath, ec);
    }

    item.teacherPayload = std::move(qltc.payload);
    item.pcmInterleaved = std::move(pcm);
    item.frames = frames;
    item.name = entry.qltcPath.filename().string();

    return true;
}

bool Trainer::preprocessWavToF32Stereo(
    const std::string& wavPath,
    const std::string& pcmPath,
    uint16_t sampleRate,
    std::string& error
) {
    std::stringstream ffm;

    ffm << "ffmpeg -hide_banner -loglevel error -y -i \"" << wavPath
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
        << " \"" << pcmPath << "\""
        << " >nul 2>&1";

    const int rc = std::system(ffm.str().c_str());

    if (rc != 0) {
        error = "ffmpeg preprocessing failed.";
        return false;
    }

    return true;
}

bool Trainer::loadF32StereoPcm(
    const std::string& pcmPath,
    std::vector<float>& interleaved,
    uint32_t& frames,
    std::string& error
) {
    std::ifstream src(pcmPath, std::ios::binary);

    if (!src) {
        error = "PCM not found.";
        return false;
    }

    src.seekg(0, std::ios::end);
    const std::streamoff bytes = src.tellg();
    src.seekg(0, std::ios::beg);

    if (bytes <= 0 || (bytes % (2 * sizeof(float))) != 0) {
        error = "PCM size invalid. Expected f32le stereo.";
        return false;
    }

    const uint64_t frameCount64 =
        static_cast<uint64_t>(bytes) / (2ULL * sizeof(float));

    if (frameCount64 > UINT32_MAX) {
        error = "PCM too large.";
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

bool Trainer::loadDatasetPayloads(
    const std::string& datasetFolder,
    std::vector<std::vector<uint8_t>>& payloads,
    std::string& error
) {
    namespace fs = std::filesystem;

    payloads.clear();

    if (!fs::exists(datasetFolder)) {
        error = "Dataset folder does not exist.";
        return false;
    }

    if (!fs::is_directory(datasetFolder)) {
        error = "Dataset path is not a folder.";
        return false;
    }

    for (const auto& entry : fs::directory_iterator(datasetFolder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const auto path = entry.path();

        if (path.extension() != ".qltc") {
            continue;
        }

        QltcFile qltc;
        std::string readError;

        if (!readQltc(path.string(), qltc, readError)) {
            std::cout << "[WARN] Skipping " << path.filename().string()
                << ": " << readError << "\n";
            continue;
        }

        payloads.push_back(std::move(qltc.payload));

        std::cout << "[DATA] Loaded " << path.filename().string() << "\n";
    }

    if (payloads.empty()) {
        error = "No valid .qltc files found in dataset folder.";
        return false;
    }

    return true;
}

uint8_t Trainer::randomValidByte() {
    std::uniform_int_distribution<size_t> dist(0, validSymbols.size() - 1);
    return validSymbols[dist(rng)];
}
