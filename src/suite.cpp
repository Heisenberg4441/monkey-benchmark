#include "suite_v1.h"

#include <algorithm>
#include <string>
#include <vector>

namespace monkey {

// Разбор эталона в индексы алфавита: уникальные UTF-8 символы, отсортированные
// побайтно. Общий для suite и custom: один и тот же текст даёт один и тот же
// алфавит и target на любой платформе.
std::vector<std::string> split_utf8(const std::string& str) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < str.size()) {
        size_t len = 1;
        const auto c = static_cast<unsigned char>(str[i]);
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        chars.push_back(str.substr(i, len));
        i += len;
    }
    return chars;
}

void reference_to_target(const std::string& text, uint32_t* n, std::vector<int32_t>* target) {
    const std::vector<std::string> chars = split_utf8(text);
    std::vector<std::string> alphabet = chars;
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
    *n = static_cast<uint32_t>(alphabet.size());
    target->clear();
    for (const auto& ch : chars)
        target->push_back(static_cast<int32_t>(std::lower_bound(alphabet.begin(), alphabet.end(), ch) -
                                               alphabet.begin()));
}

namespace suite_v1 {

KernelParams make_params(const TestDef& t, Category c) {
    KernelParams p;
    p.kernel = t.kernel;
    switch (t.kernel) {
    case KernelId::MonkeyRandom:
    case KernelId::MonkeyBrute:
        p.seed = kSeed;
        reference_to_target(kFox, &p.n, &p.target);
        break;
    case KernelId::Bbp:
        p.bbp = bbp_window(static_cast<uint32_t>(t.size[c]), kBbpBlockTerms);
        break;
    case KernelId::MillerRabin:
        p.mr_base = kMrBase;
        p.mr_step = kMrStep;
        break;
    }
    return p;
}

uint64_t work_units(const TestDef& t, Category c) {
    return t.kernel == KernelId::Bbp ? bbp_units(make_params(t, c).bbp) : t.size[c];
}

const char* category_key(Category c) {
    switch (c) {
    case kCpuSt: return "cpu_st";
    case kCpuMt: return "cpu_mt";
    default: return "gpu";
    }
}

const char* category_title(Category c) {
    switch (c) {
    case kCpuSt: return "CPU single-thread";
    case kCpuMt: return "CPU multi-thread";
    default: return "GPU";
    }
}

} // namespace suite_v1
} // namespace monkey
