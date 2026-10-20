#include "report.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "json.h"
#include "suite_v1.h"

namespace monkey {

namespace {

const char* column_title(const std::string& key) {
    if (key == "cpu_st") return "CPU-ST";
    if (key == "cpu_mt") return "CPU-MT";
    if (key == "gpu_vulkan") return "GPU Vulkan";
    if (key == "gpu_cuda") return "GPU CUDA";
    if (key == "combined") return "CPU+GPU";
    return key.c_str();
}

// Порядок колонок таблицы и категорий в JSON.
const char* const kOrder[] = {"cpu_st", "cpu_mt", "gpu_cuda", "gpu_vulkan", "combined", "custom"};

// Выравнивание по числу символов, а не байт: «±» в UTF-8 занимает два байта.
std::string pad_left(const std::string& s, size_t width) {
    size_t chars = 0;
    for (unsigned char c : s) chars += (c & 0xC0) != 0x80;
    return chars >= width ? s : std::string(width - chars, ' ') + s;
}

std::string fmt_score(double s) {
    if (!std::isfinite(s)) return "-";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f", s);
    return buf;
}

} // namespace

double geomean(const std::vector<double>& v) {
    if (v.empty()) return kNaN;
    double log_sum = 0;
    for (double x : v) {
        if (!std::isfinite(x) || x <= 0) return kNaN;
        log_sum += std::log(x);
    }
    return std::exp(log_sum / static_cast<double>(v.size()));
}

void compute_scores(Report& r) {
    std::map<std::string, std::vector<double>> by_cat;
    std::map<std::string, int> counts;
    for (const TestOutcome& t : r.tests) {
        r.invalid |= !t.valid;
        r.unstable |= t.unstable;
        by_cat[t.category].push_back(t.valid ? t.score : kNaN);
        ++counts[t.category];
    }
    for (const auto& [cat, scores] : by_cat) {
        if (cat == "combined" || cat == "custom") continue; // информативно, без балла
        // Категория считается только по полному набору тестов suite.
        r.category_scores[cat] =
            counts[cat] == suite_v1::kTestCount ? geomean(scores) : kNaN;
    }
    auto get = [&](const char* k) {
        const auto it = r.category_scores.find(k);
        return it == r.category_scores.end() ? kNaN : it->second;
    };
    const double st = get("cpu_st"), mt = get("cpu_mt");
    if (std::isfinite(st) && std::isfinite(mt)) r.cpu_score = geomean({st, mt});
    for (const char* k : {"gpu_cuda", "gpu_vulkan"}) {
        const double g = get(k);
        if (std::isfinite(g) && (!std::isfinite(r.gpu_score) || g > r.gpu_score)) {
            r.gpu_score = g;
            r.gpu_best = k;
        }
    }
    if (std::isfinite(st) && std::isfinite(mt) && std::isfinite(r.gpu_score))
        r.overall = geomean({st, mt, r.gpu_score});
    r.partial = !std::isfinite(r.overall);
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string report_json(const Report& r) {
    JsonWriter w;
    w.begin_object();
    w.kv("schema_version", kSchemaVersion);
    w.kv("suite_version", suite_v1::kVersion);
    w.kv("mode", r.mode);
    w.kv("runs", r.runs);
    w.kv("timestamp", r.timestamp);
    w.kv("reference_machine", suite_v1::kReferenceMachine);

    w.key("binary").begin_object();
    w.kv("version", r.build.version);
    w.kv("git_sha", r.build.git_sha);
    w.kv("compiler", r.build.compiler);
    w.kv("flags", r.build.flags);
    w.kv("build_type", r.build.build_type);
    w.kv("arith", r.build.portable_arith ? "portable" : "native");
    w.end_object();

    w.key("system").begin_object();
    w.kv_opt("os", r.sys.os);
    w.kv_opt("os_version", r.sys.os_version);
    w.kv_opt("kernel", r.sys.kernel);
    w.kv_opt("arch", r.sys.arch);
    w.kv_opt("cpu_model", r.sys.cpu_model);
    w.kv_opt("cores_physical", uint64_t{r.sys.cores_physical});
    w.kv_opt("cores_logical", uint64_t{r.sys.cores_logical});
    w.kv_opt("cores_performance", uint64_t{r.sys.cores_performance});
    w.kv_opt("cores_efficiency", uint64_t{r.sys.cores_efficiency});
    w.kv_opt("ram_bytes", r.sys.ram_bytes);
    w.end_object();

    w.key("devices").begin_array();
    for (const DeviceRow& d : r.devices) {
        w.begin_object();
        w.kv("api", d.api);
        w.kv_opt("name", d.name);
        w.kv_opt("vendor", d.vendor);
        w.kv_opt("driver", d.driver);
        w.kv_opt("compute_units", uint64_t{d.compute_units});
        w.kv_opt("u64_mode", d.u64_mode);
        w.kv("init_time_s", d.init_time_s);
        w.end_object();
    }
    w.end_array();

    w.key("tests").begin_array();
    for (const TestOutcome& t : r.tests) {
        w.begin_object();
        w.kv("name", t.name);
        w.kv("category", t.category);
        w.kv("backend", t.backend);
        w.kv("work_units", t.work_units);
        w.key("runs").begin_array();
        for (double x : t.runs) w.value(x);
        w.end_array();
        if (t.runs.empty()) {
            w.key("time_median").null();
        } else {
            w.kv("time_median", t.time_median);
        }
        w.kv("time_min", t.runs.empty() ? kNaN : t.time_min);
        w.kv("time_max", t.runs.empty() ? kNaN : t.time_max);
        w.kv("cv", t.runs.size() > 1 ? t.cv : kNaN);
        w.kv("ops_per_sec", t.ops_per_sec);
        w.key("checksum").hex(t.checksum);
        if (t.has_expected) {
            w.key("expected_checksum").hex(t.expected_checksum);
            w.kv("hits", t.hits);
            w.kv("expected_hits", t.expected_hits);
        } else {
            w.key("expected_checksum").null();
            w.kv("hits", t.hits);
            w.key("expected_hits").null();
        }
        w.kv("valid", t.valid);
        w.kv_opt("invalid_reason", t.invalid_reason);
        w.kv("unstable", t.unstable);
        w.kv("t_ref", t.t_ref > 0 ? t.t_ref : kNaN);
        w.kv("score", t.score);
        w.kv("warmup_s", t.warmup_s);
        w.kv("chunk_size", t.chunk);
        w.kv("cpu_units", t.cpu_units);
        w.kv("gpu_units", t.gpu_units);
        if (t.timed) {
            w.key("timed").begin_object();
            w.kv("seconds", t.seconds);
            w.kv("passes", t.passes);
            w.kv("found", t.found);
            if (t.first_full != UINT64_MAX) w.kv("first_match_index", t.first_full);
            w.key("throughput").begin_object();
            w.kv("samples", uint64_t{t.throughput.n});
            w.kv("min", t.throughput.min);
            w.kv("median", t.throughput.median);
            w.kv("mean", t.throughput.mean);
            w.kv("p95", t.throughput.p95);
            w.kv("max", t.throughput.max);
            w.kv("cv", t.throughput.cv);
            w.end_object();
            w.kv("throttle_ratio", t.throttle_ratio);
            w.end_object();
        }
        w.end_object();
    }
    w.end_array();

    w.key("scores").begin_object();
    w.key("categories").begin_object();
    for (const char* k : kOrder) {
        const auto it = r.category_scores.find(k);
        if (it != r.category_scores.end()) w.kv(k, it->second);
    }
    w.end_object();
    w.kv("cpu", r.cpu_score);
    w.kv("gpu", r.gpu_score);
    w.kv_opt("gpu_api", r.gpu_best.empty() ? std::string() : r.gpu_best.substr(4));
    w.kv("overall", r.overall);
    w.end_object();

    w.key("flags").begin_object();
    w.kv("comparable", !r.custom && !r.build.debug && !r.build.native_arch && r.mode != "stress");
    w.kv("custom", r.custom);
    w.kv("debug_build", r.build.debug);
    w.kv("native_arch", r.build.native_arch);
    w.kv("portable_arith", r.build.portable_arith);
    w.kv("invalid", r.invalid);
    w.kv("unstable", r.unstable);
    w.kv("partial", r.partial);
    w.end_object();
    w.end_object();
    return w.str();
}

void print_scores(const Report& r, std::ostream& os) {
    std::vector<std::string> cols;
    for (const char* k : kOrder) {
        for (const TestOutcome& t : r.tests) {
            if (t.category == k) {
                cols.emplace_back(k);
                break;
            }
        }
    }
    os << "\nScores — suite v" << suite_v1::kVersion << ", " << r.mode << " (" << r.runs
       << (r.runs == 1 ? " run" : " runs") << " per test, median), 1000 = " << suite_v1::kReferenceMachine
       << "\n\n";
    os << std::left << std::setw(16) << "Test";
    for (const auto& c : cols) os << std::right << std::setw(16) << column_title(c);
    os << "\n";
    for (int i = 0; i < suite_v1::kTestCount; ++i) {
        const char* name = suite_v1::kTests[i].name;
        bool any = false;
        for (const TestOutcome& t : r.tests) any |= t.name == name;
        if (!any) continue;
        os << std::left << std::setw(16) << name;
        for (const auto& c : cols) {
            std::string cell = "";
            for (const TestOutcome& t : r.tests) {
                if (t.name != name || t.category != c) continue;
                if (!t.valid) {
                    cell = "INVALID";
                } else if (std::isfinite(t.score)) {
                    char buf[48];
                    if (t.runs.size() > 1)
                        std::snprintf(buf, sizeof buf, "%.0f ±%.1f%%%s", t.score, t.cv * 100, t.unstable ? "*" : "");
                    else
                        std::snprintf(buf, sizeof buf, "%.0f", t.score);
                    cell = buf;
                } else {
                    char buf[32];
                    std::snprintf(buf, sizeof buf, "%.2f s", t.time_median);
                    cell = buf;
                }
            }
            os << pad_left(cell, 16);
        }
        os << "\n";
    }
    os << std::left << std::setw(16) << "Category";
    for (const auto& c : cols) {
        const auto it = r.category_scores.find(c);
        os << std::right << std::setw(16) << (it == r.category_scores.end() ? "-" : fmt_score(it->second));
    }
    os << "\n\n";
    if (r.unstable) os << "* CV of run times above " << suite_v1::kUnstableCv * 100 << "% — result unstable\n";
    os << "CPU score:      " << std::setw(6) << fmt_score(r.cpu_score) << "   geomean(CPU-ST, CPU-MT)\n";
    if (!r.gpu_best.empty())
        os << "GPU score:      " << std::setw(6) << fmt_score(r.gpu_score) << "   best API: "
           << column_title(r.gpu_best) << "\n";
    os << "Overall score:  " << std::setw(6) << fmt_score(r.overall) << "   geomean(CPU-ST, CPU-MT, GPU)";
    if (r.partial) os << " — not computed (needs CPU-ST, CPU-MT and a GPU)";
    os << "\n";
    if (r.invalid) os << "\nINVALID: a checksum did not match its golden value; no comparable score.\n";
}

} // namespace monkey
