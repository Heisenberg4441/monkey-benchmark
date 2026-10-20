// Движок исполнения (engine.h): fixed-work даёт тот же результат, что эталон,
// при любом числе потоков; общий диспетчер CPU+GPU не дублирует работу; стоп по
// времени и по совпадению срабатывает быстро; ошибка устройства останавливает
// задание с текстом ошибки, а не вешает его.
#include "catch.hpp"
#include "engine.h"
#include "runner.h"
#include "kernels.h"

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace monkey;

namespace {

// «GPU», который считает на CPU тем же эталоном и умеет падать по команде.
class FakeGpu final : public GpuDevice {
public:
    explicit FakeGpu(int fail_after = -1) : fail_after_(fail_after) { info_.api = "fake"; }
    const GpuInfo& info() const override { return info_; }
    void prepare(const KernelParams& p) override {
        p_ = p;
        reset_bbp_acc();
    }
    RangeResult run_range(uint64_t b, uint64_t c) override {
        if (fail_after_ >= 0 && calls_++ >= fail_after_) throw GpuError("fake: device lost");
        units += c;
        return cpu_run_range(p_, b, c, acc_.data());
    }
    void read_bbp_acc(std::vector<uint64_t>& a) override { a = acc_; }
    void reset_bbp_acc() override { acc_.assign(p_.bbp.positions + 1, 0); }
    std::atomic<uint64_t> units{0};

private:
    GpuInfo info_;
    KernelParams p_;
    std::vector<uint64_t> acc_;
    int fail_after_;
    int calls_ = 0;
};

KernelParams random_fox() {
    KernelParams p;
    p.kernel = KernelId::MonkeyRandom;
    p.n = 28;
    for (int i = 0; i < 43; ++i) p.target.push_back((i * 7) % 28);
    return p;
}

KernelParams small_bbp() {
    KernelParams p;
    p.kernel = KernelId::Bbp;
    p.bbp.pos0 = 9000;
    p.bbp.positions = 6;
    p.bbp.block_terms = 256;
    p.bbp.blocks = (9000 + 5 + 256) / 256;
    return p;
}

} // namespace

TEST_CASE("fixed-work result does not depend on thread count", "[engine]") {
    const KernelParams kp = random_fox();
    const uint64_t N = 2000000;
    const RangeResult ref = cpu_run_range(kp, 0, N, nullptr);
    for (unsigned threads : {1u, 3u, 8u}) {
        Engine e(threads, nullptr);
        e.plan.cpu = 4099;
        JobSpec s;
        s.kp = &kp;
        s.end = N;
        const JobResult r = e.run(s);
        INFO("threads " << threads);
        REQUIRE(r.error.empty());
        REQUIRE(r.completed);
        REQUIRE(r.res == ref);
        REQUIRE(r.seconds > 0);
    }
}

TEST_CASE("CPU and GPU share one dispatcher without duplicates", "[engine]") {
    const KernelParams kp = small_bbp();
    const uint64_t N = bbp_units(kp.bbp);
    std::vector<uint64_t> ref_acc(kp.bbp.positions, 0);
    const RangeResult ref = cpu_run_range(kp, 0, N, ref_acc.data());
    FakeGpu gpu;
    Engine e(4, &gpu);
    e.plan.cpu = 3;
    e.plan.gpu = 5;
    JobSpec s;
    s.kp = &kp;
    s.end = N;
    s.use_gpu = true;
    const JobResult r = e.run(s);
    REQUIRE(r.error.empty());
    REQUIRE(r.res == ref);
    REQUIRE(r.cpu_units + r.gpu_units == N);
    REQUIRE(r.gpu_units > 0);
    REQUIRE(r.cpu_units > 0);
    std::vector<uint64_t> got = r.bbp_acc;
    got.resize(kp.bbp.positions);
    REQUIRE(got == ref_acc);
}

TEST_CASE("time limit stops a job within about one chunk", "[engine]") {
    const KernelParams kp = random_fox();
    Engine e(4, nullptr);
    JobSpec s;
    s.kp = &kp;
    s.end = UINT64_MAX;
    s.max_seconds = 0.3;
    s.adaptive = true; // чанки ~15 мс
    const auto t0 = std::chrono::steady_clock::now();
    const JobResult r = e.run(s);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(r.error.empty());
    REQUIRE_FALSE(r.completed);
    REQUIRE(r.res.count > 0);
    REQUIRE(wall < 0.6);
    REQUIRE(e.plan.cpu > 0);
}

TEST_CASE("stop on full match", "[engine]") {
    KernelParams kp;
    kp.kernel = KernelId::MonkeyBrute;
    kp.n = 10;
    kp.target = {9, 8, 7, 6, 5, 4, 3, 2}; // код 98765432
    Engine e(4, nullptr);
    e.plan.cpu = 1u << 16;
    JobSpec s;
    s.kp = &kp;
    s.end = UINT64_MAX;
    s.stop_on_full_match = true;
    const JobResult r = e.run(s);
    REQUIRE(r.error.empty());
    REQUIRE(r.res.first_full == 98765432ull);
    REQUIRE(r.res.count < 98765432ull + 4 * (1u << 17));
}

TEST_CASE("a failing device stops the job with its error", "[engine]") {
    const KernelParams kp = random_fox();
    FakeGpu gpu(3);
    Engine e(2, &gpu);
    JobSpec s;
    s.kp = &kp;
    s.end = UINT64_MAX; // без ошибки шло бы бесконечно
    s.use_gpu = true;
    const JobResult r = e.run(s);
    REQUIRE_FALSE(r.completed);
    REQUIRE(r.error == "fake: device lost");
    // Пул пережил ошибку: следующее задание работает.
    JobSpec ok;
    ok.kp = &kp;
    ok.end = 100000;
    const JobResult r2 = e.run(ok);
    REQUIRE(r2.completed);
    REQUIRE(r2.res == cpu_run_range(kp, 0, 100000, nullptr));
}

TEST_CASE("chunk samples cover all processed units", "[engine]") {
    const KernelParams kp = random_fox();
    Engine e(3, nullptr);
    e.plan.cpu = 10007;
    JobSpec s;
    s.kp = &kp;
    s.end = 1000000;
    s.keep_samples = true;
    const JobResult r = e.run(s);
    uint64_t sum = 0;
    for (const auto& c : r.samples) {
        sum += c.units;
        REQUIRE(c.t_end >= c.t_begin);
        REQUIRE(c.t_end <= r.seconds + 1e-9);
    }
    REQUIRE(sum == 1000000);
}

TEST_CASE("fixed-work runner: warmup tunes chunks, repeats are identical", "[engine][runner]") {
    const KernelParams kp = small_bbp();
    const uint64_t N = bbp_units(kp.bbp);
    Engine e(4, nullptr);
    const FixedRun fr = run_fixed(e, kp, N, DeviceSel{}, 0.2, 3);
    REQUIRE(fr.error.empty());
    REQUIRE(fr.times.size() == 3);
    REQUIRE(fr.warmup_s >= 0.2);
    REQUIRE(fr.plan.cpu > 0);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(fr.results[i] == fr.results[0]);
        REQUIRE(fr.results[i].count == N);
        REQUIRE(bbp_verify(kp.bbp, fr.bbp_acc[i]).bad_positions == 0);
    }
}

TEST_CASE("timed runner repeats passes over a window until the time is up", "[engine][runner]") {
    const KernelParams kp = small_bbp();
    const uint64_t N = bbp_units(kp.bbp);
    Engine e(4, nullptr);
    const TimedRun tr = run_timed(e, kp, N, 0.5, false, DeviceSel{}, 0.0);
    REQUIRE(tr.error.empty());
    REQUIRE(tr.passes >= 1);
    REQUIRE(tr.seconds >= 0.45);
    REQUIRE(tr.seconds < 0.8);
    const auto series = throughput_series(tr.samples, 0.1, tr.seconds);
    REQUIRE(series.size() >= 4);
    double units = 0;
    for (const auto& c : tr.samples) units += static_cast<double>(c.units);
    REQUIRE(units == static_cast<double>(tr.res.count));
}

TEST_CASE("series statistics", "[runner]") {
    const SeriesStats s = series_stats({4, 1, 3, 2});
    REQUIRE(s.n == 4);
    REQUIRE(s.min == 1);
    REQUIRE(s.max == 4);
    REQUIRE(s.median == 2.5);
    REQUIRE(s.mean == 2.5);
    REQUIRE(std::abs(s.cv - std::sqrt(5.0 / 3.0) / 2.5) < 1e-12);
}

TEST_CASE("a range ending at the top of the 64-bit space terminates", "[engine]") {
    // Регрессия: fetch_add диспетчера переполнялся у 2^64 и уходил по кругу.
    KernelParams kp;
    kp.kernel = KernelId::MonkeyBrute;
    kp.n = 28;
    kp.target = {1, 2, 3, 4, 5};
    Engine e(4, nullptr);
    e.plan.cpu = 1u << 18;
    JobSpec s;
    s.kp = &kp;
    s.begin = ~0ull - (1ull << 20);
    s.end = ~0ull;
    const JobResult r = e.run(s);
    REQUIRE(r.completed);
    REQUIRE(r.res == cpu_run_range(kp, s.begin, 1ull << 20, nullptr));
}
