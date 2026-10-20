#include "engine.h"

#include <algorithm>
#include <chrono>

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace monkey {

namespace {

using clk = std::chrono::steady_clock;

double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

// macOS: высший QoS, чтобы планировщик держал потоки бенча на P-ядрах, пока
// хватает P-ядер (CPU-ST — всегда). На Linux/Windows привязки нет.
void set_thread_qos() {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

} // namespace

uint64_t initial_chunk(KernelId k, bool gpu) {
    switch (k) {
    case KernelId::MonkeyRandom: return gpu ? 1ull << 22 : 1ull << 16;
    case KernelId::MonkeyBrute: return gpu ? 1ull << 23 : 1ull << 18;
    case KernelId::Bbp: return gpu ? 1ull << 13 : 4;
    case KernelId::MillerRabin: return gpu ? 1ull << 18 : 1ull << 12;
    }
    return 1024;
}

struct Engine::Job {
    JobSpec spec;
    unsigned cpu_workers = 0;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> abort{false};
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    clk::time_point t0;
    uint64_t cpu_chunk0 = 0, gpu_chunk0 = 0;
    // Выход по воркерам: каждый пишет только свой слот.
    std::vector<RangeResult> res;
    std::vector<std::vector<uint64_t>> bbp;
    std::vector<std::vector<ChunkSample>> samples;
    std::vector<double> finish;
    std::vector<uint64_t> final_chunk;
    std::vector<uint8_t> took_part;
    AttemptCounters progress;
    std::mutex err_m;
    std::string error;

    bool participates(unsigned id, unsigned cpu_threads, bool has_gpu) const {
        if (id < cpu_threads) return spec.use_cpu && id < cpu_workers;
        return spec.use_gpu && has_gpu;
    }
};

Engine::Engine(unsigned cpu_threads, GpuDevice* gpu) : cpu_threads_(cpu_threads), gpu_(gpu) {
    const unsigned n = cpu_threads_ + (gpu_ ? 1 : 0);
    threads_.reserve(n);
    for (unsigned i = 0; i < n; ++i) threads_.emplace_back(&Engine::worker_main, this, i);
}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lk(m_);
        quit_ = true;
    }
    cv_job_.notify_all();
    for (auto& t : threads_) t.join();
}

void Engine::worker_main(unsigned id) {
    set_thread_qos();
    uint64_t seen = 0;
    for (;;) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_job_.wait(lk, [&] { return quit_ || generation_ != seen; });
            if (quit_) return;
            seen = generation_;
            j = job_;
        }
        if (j->participates(id, cpu_threads_, gpu_ != nullptr)) {
            j->took_part[id] = 1;
            j->ready.fetch_add(1, std::memory_order_acq_rel);
            while (!j->go.load(std::memory_order_acquire)) std::this_thread::yield();
            work(id, *j);
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) cv_done_.notify_all();
        }
    }
}

void Engine::work(unsigned id, Job& j) {
    const bool is_gpu = id == cpu_threads_;
    const KernelParams& kp = *j.spec.kp;
    const bool is_bbp = kp.kernel == KernelId::Bbp;
    const double target = is_gpu ? gpu_target_s : cpu_target_s;
    uint64_t chunk = is_gpu ? j.gpu_chunk0 : j.cpu_chunk0;

    RangeResult acc;
    std::vector<uint64_t> bbp;
    if (is_bbp && !is_gpu) bbp.assign(kp.bbp.positions, 0);
    std::vector<ChunkSample> samples;

    try {
        while (!j.abort.load(std::memory_order_relaxed)) {
            if (j.spec.max_seconds > 0 && since(j.t0) >= j.spec.max_seconds) {
                j.abort.store(true, std::memory_order_relaxed);
                break;
            }
            // CAS вместо fetch_add: счётчик не уходит за end и не переполняется у
            // конца 64-битного пространства (fetch_add там заворачивался в 0).
            uint64_t b = j.next.load(std::memory_order_relaxed), c = 0;
            do {
                if (b >= j.spec.end) break;
                c = std::min(chunk, j.spec.end - b);
            } while (!j.next.compare_exchange_weak(b, b + c, std::memory_order_relaxed));
            if (b >= j.spec.end) break;
            const double ta = since(j.t0);
            const RangeResult r = is_gpu ? gpu_->run_range(b, c)
                                         : cpu_run_range(kp, b, c, is_bbp ? bbp.data() : nullptr);
            const double tb = since(j.t0);
            acc.merge(r);
            j.progress.record(id, acc.count);
            if (j.spec.keep_samples) samples.push_back({ta, tb, c, id});
            if (j.spec.stop_on_full_match && r.first_full != UINT64_MAX)
                j.abort.store(true, std::memory_order_relaxed);
            if (j.spec.adaptive) {
                const double scale = std::clamp(target / std::max(tb - ta, 1e-6), 0.25, 4.0);
                chunk = std::max<uint64_t>(1, static_cast<uint64_t>(static_cast<double>(chunk) * scale));
            }
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(j.err_m);
        if (j.error.empty()) j.error = e.what();
        j.abort.store(true, std::memory_order_relaxed);
    }
    j.res[id] = acc;
    j.bbp[id] = std::move(bbp);
    j.samples[id] = std::move(samples);
    j.final_chunk[id] = chunk;
    j.finish[id] = since(j.t0);
}

JobResult Engine::run(const JobSpec& spec, const std::function<void(uint64_t, double)>& progress) {
    JobResult out;
    Job j;
    j.spec = spec;
    j.cpu_workers = (spec.cpu_workers == 0) ? cpu_threads_ : std::min(spec.cpu_workers, cpu_threads_);
    j.next.store(spec.begin);
    const KernelId k = spec.kp->kernel;
    j.cpu_chunk0 = plan.cpu ? plan.cpu : initial_chunk(k, false);
    j.gpu_chunk0 = plan.gpu ? plan.gpu : initial_chunk(k, true);
    const unsigned n = static_cast<unsigned>(threads_.size());
    j.res.resize(n);
    j.bbp.resize(n);
    j.samples.resize(n);
    j.finish.assign(n, 0);
    j.final_chunk.assign(n, 0);
    j.took_part.assign(n, 0);
    j.progress.resize(n);

    const bool gpu_on = spec.use_gpu && gpu_;
    if (gpu_on) {
        try {
            gpu_->prepare(*spec.kp);
        } catch (const std::exception& e) {
            out.error = e.what();
            return out;
        }
    }
    unsigned participants = 0;
    for (unsigned id = 0; id < n; ++id) participants += j.participates(id, cpu_threads_, gpu_ != nullptr);
    if (participants == 0) {
        out.error = "engine: no devices selected";
        return out;
    }

    {
        std::lock_guard<std::mutex> lk(m_);
        job_ = &j;
        pending_ = n;
        ++generation_;
    }
    cv_job_.notify_all();
    while (j.ready.load(std::memory_order_acquire) < participants) std::this_thread::yield();
    j.t0 = clk::now();
    j.go.store(true, std::memory_order_release);

    {
        std::unique_lock<std::mutex> lk(m_);
        while (pending_ != 0) {
            cv_done_.wait_for(lk, std::chrono::milliseconds(250));
            if (pending_ != 0 && progress) {
                lk.unlock();
                progress(j.progress.total(), since(j.t0));
                lk.lock();
            }
        }
        job_ = nullptr;
    }

    // Слияние: порядок не важен — все поля складываются.
    if (k == KernelId::Bbp) out.bbp_acc.assign(spec.kp->bbp.positions, 0);
    std::vector<uint64_t> cpu_chunks;
    for (unsigned id = 0; id < n; ++id) {
        if (!j.took_part[id]) continue;
        out.res.merge(j.res[id]);
        out.seconds = std::max(out.seconds, j.finish[id]);
        (id < cpu_threads_ ? out.cpu_units : out.gpu_units) += j.res[id].count;
        for (size_t w = 0; w < j.bbp[id].size(); ++w) out.bbp_acc[w] += j.bbp[id][w];
        out.samples.insert(out.samples.end(), j.samples[id].begin(), j.samples[id].end());
        if (id < cpu_threads_) cpu_chunks.push_back(j.final_chunk[id]);
    }
    std::sort(out.samples.begin(), out.samples.end(),
              [](const ChunkSample& a, const ChunkSample& b) { return a.t_end < b.t_end; });
    out.error = j.error;
    if (gpu_on && out.error.empty() && k == KernelId::Bbp) {
        try {
            std::vector<uint64_t> g;
            gpu_->read_bbp_acc(g);
            for (size_t w = 0; w < g.size() && w < out.bbp_acc.size(); ++w) out.bbp_acc[w] += g[w];
        } catch (const std::exception& e) {
            out.error = e.what();
        }
    }
    // Итог подстройки: медиана CPU-потоков (P- и E-ядра различаются) и чанк GPU.
    if (spec.adaptive && out.error.empty()) {
        if (!cpu_chunks.empty()) {
            std::nth_element(cpu_chunks.begin(), cpu_chunks.begin() + cpu_chunks.size() / 2, cpu_chunks.end());
            plan.cpu = cpu_chunks[cpu_chunks.size() / 2];
        }
        if (gpu_on) plan.gpu = j.final_chunk[cpu_threads_];
    }
    out.completed = out.error.empty() && out.res.count == spec.end - spec.begin;
    return out;
}

} // namespace monkey
