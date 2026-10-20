#include "runner.h"

#include <algorithm>
#include <cmath>

namespace monkey {

SeriesStats series_stats(std::vector<double> v) {
    SeriesStats s;
    s.n = v.size();
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.min = v.front();
    s.max = v.back();
    s.median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    s.p95 = v[std::min(n - 1, static_cast<size_t>(n * 0.95))];
    s.p99 = v[std::min(n - 1, static_cast<size_t>(n * 0.99))];
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / static_cast<double>(n);
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    // Выборочное стандартное отклонение (n - 1): прогонов обычно мало.
    s.stddev = n > 1 ? std::sqrt(sq / static_cast<double>(n - 1)) : 0;
    s.cv = s.mean > 0 ? s.stddev / s.mean : 0;
    return s;
}

std::vector<double> throughput_series(const std::vector<ChunkSample>& samples, double bin_s,
                                      double total_s) {
    const auto bins = static_cast<size_t>(total_s / bin_s);
    std::vector<double> units(bins, 0.0);
    for (const ChunkSample& c : samples) {
        const double len = std::max(c.t_end - c.t_begin, 1e-12);
        const double rate = static_cast<double>(c.units) / len;
        auto b = static_cast<size_t>(c.t_begin / bin_s);
        for (; b < bins; ++b) {
            const double lo = std::max(c.t_begin, b * bin_s);
            const double hi = std::min(c.t_end, (b + 1) * bin_s);
            if (hi <= lo) break;
            units[b] += rate * (hi - lo);
        }
    }
    for (double& u : units) u /= bin_s;
    return units;
}

namespace {

JobSpec make_spec(const KernelParams& kp, const DeviceSel& sel) {
    JobSpec s;
    s.kp = &kp;
    s.use_cpu = sel.use_cpu;
    s.cpu_workers = sel.cpu_workers;
    s.use_gpu = sel.use_gpu;
    return s;
}

} // namespace

FixedRun run_fixed(Engine& e, const KernelParams& kp, uint64_t N, const DeviceSel& sel,
                   double warmup_s, int repeats, const Progress& progress) {
    FixedRun out;
    // Прогрев в той же сессии: частоты выходят на рабочие, чанк подстраивается
    // под цель. Диапазон — тот же тест, проходы по кругу до истечения времени.
    e.plan = ChunkPlan{};
    double warm = 0;
    do {
        JobSpec s = make_spec(kp, sel);
        s.end = N;
        s.adaptive = true;
        s.max_seconds = std::max(warmup_s - warm, 0.05);
        const JobResult r = e.run(s, [&](uint64_t d, double t) {
            if (progress) progress("warmup", d, N, warm + t);
        });
        if (!r.error.empty()) {
            out.error = r.error;
            return out;
        }
        warm += r.seconds;
    } while (warm < warmup_s);
    out.warmup_s = warm;
    out.plan = e.plan;

    for (int i = 0; i < repeats; ++i) {
        JobSpec s = make_spec(kp, sel);
        s.end = N;
        const JobResult r = e.run(s, [&](uint64_t d, double t) {
            if (progress) progress("run", d, N, t);
        });
        if (!r.error.empty()) {
            out.error = r.error;
            return out;
        }
        out.times.push_back(r.seconds);
        out.results.push_back(r.res);
        out.bbp_acc.push_back(r.bbp_acc);
        out.cpu_units = r.cpu_units;
        out.gpu_units = r.gpu_units;
    }
    return out;
}

TimedRun run_timed(Engine& e, const KernelParams& kp, uint64_t range_end, double seconds,
                   bool stop_on_match, const DeviceSel& sel, double warmup_s,
                   const Progress& progress, bool tune) {
    TimedRun out;
    if (tune) e.plan = ChunkPlan{};
    if (tune && warmup_s > 0) {
        JobSpec s = make_spec(kp, sel);
        s.end = range_end;
        s.adaptive = true;
        s.max_seconds = warmup_s;
        const JobResult r = e.run(s);
        if (!r.error.empty()) {
            out.error = r.error;
            return out;
        }
    } else {
        // Без прогрева чанк подстраивается прямо в замере.
    }
    double elapsed = 0;
    for (;;) {
        JobSpec s = make_spec(kp, sel);
        s.end = range_end;
        s.adaptive = tune && warmup_s <= 0;
        s.keep_samples = true;
        s.stop_on_full_match = stop_on_match;
        if (seconds > 0) s.max_seconds = seconds - elapsed;
        const double offset = elapsed;
        const JobResult r = e.run(s, [&](uint64_t d, double t) {
            if (progress) progress("timed", out.res.count + d, 0, offset + t);
        });
        if (!r.error.empty()) {
            out.error = r.error;
            return out;
        }
        for (ChunkSample c : r.samples) {
            c.t_begin += offset;
            c.t_end += offset;
            out.samples.push_back(c);
        }
        out.res.merge(r.res);
        out.cpu_units += r.cpu_units;
        out.gpu_units += r.gpu_units;
        elapsed += r.seconds;
        if (r.res.first_full != UINT64_MAX) out.found = true;
        if (r.completed) {
            ++out.passes;
            out.pass_results.push_back(r.res);
            out.pass_bbp.push_back(r.bbp_acc);
        }
        if (out.found || seconds <= 0 || elapsed >= seconds || !r.completed) break;
    }
    out.seconds = elapsed;
    return out;
}

} // namespace monkey
