#include "gpu_device.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "workloads/arith.h"
#include "workloads/bbp_kernel.h"
#include "workloads/miller_rabin_kernel.h"
#include "workloads/monkey_kernel.h"

// CUDA-исполнитель диапазонов. Ядра — те же header-only функции
// (__host__ __device__), что исполняет CPU, поэтому кандидат и результат
// индекса совпадают бит-в-бит. Обход и редукция повторяют Vulkan-шейдер:
// поток обрабатывает begin + gid + step·total, блок редуцирует
// {checksum, hits, count, first_full} в shared memory и пишет один слот.

namespace monkey {

namespace {

constexpr int kBlock = 256;
constexpr uint64_t kThreadsTarget = 1u << 20;

// POD-копия KernelParams для передачи в ядро по значению.
struct DevParams {
    uint32_t seed;
    uint32_t n;
    uint32_t len;
    uint64_t mr_base;
    uint64_t mr_step;
    kernel::BbpParams bbp;
};

struct Partial {
    unsigned long long checksum, hits, count, first_full;
};

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw GpuError(std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

template <int K>
__global__ void range_kernel(DevParams p, const int32_t* __restrict__ target, uint64_t begin,
                             uint64_t count, uint64_t steps, Partial* out,
                             unsigned long long* bbp_acc) {
    __shared__ unsigned long long s_cs[kBlock], s_hits[kBlock], s_cnt[kBlock], s_ff[kBlock];
    const uint64_t total = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t gid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    unsigned long long cs = 0, hits = 0, cnt = 0, ff = ~0ull;
    if constexpr (K == 3) {
        // Miller-Rabin — пошаговый автомат (MrLane): поток, закончивший число,
        // сразу берёт следующее и не ждёт соседей по warp'у с простыми.
        kernel::MrLane L{};
        uint64_t st = 0, i = begin + gid;
        bool busy = steps > 0 && gid < count;
        int res = busy ? kernel::mr_lane_start(L, p.mr_base + p.mr_step * i) : kernel::kMrContinue;
        while (busy) {
            if (res == kernel::kMrContinue) res = kernel::mr_lane_step(L);
            if (res != kernel::kMrContinue) {
                ++cnt;
                if (res == kernel::kMrPrime) {
                    ++hits;
                    cs += kernel::contrib(i, 1);
                }
                const uint64_t off = ++st * total + gid;
                busy = st < steps && off < count;
                if (busy) {
                    i = begin + off;
                    res = kernel::mr_lane_start(L, p.mr_base + p.mr_step * i);
                }
            }
        }
    }
    for (uint64_t st = 0; K != 3 && st < steps; ++st) {
        const uint64_t off = st * total + gid;
        if (off >= count) break;
        const uint64_t i = begin + off;
        uint64_t r = 0;
        if constexpr (K == 0) {
            const uint32_t v = kernel::monkey_random_prefix(p.seed, i, target, p.len, p.n);
            r = v;
            if (v == p.len && i < ff) ff = i;
        } else if constexpr (K == 1) {
            const uint32_t v = kernel::monkey_brute_suffix(i, target, p.len, p.n);
            r = v;
            if (v == p.len && i < ff) ff = i;
        } else if constexpr (K == 2) {
            r = kernel::bbp_unit(p.bbp, i);
            atomicAdd(&bbp_acc[kernel::bbp_unit_position(p.bbp, i)], static_cast<unsigned long long>(r));
        }
        ++cnt;
        if (r != 0) {
            ++hits;
            cs += kernel::contrib(i, r);
        }
    }

    const unsigned t = threadIdx.x;
    s_cs[t] = cs;
    s_hits[t] = hits;
    s_cnt[t] = cnt;
    s_ff[t] = ff;
    __syncthreads();
    for (unsigned half = kBlock / 2; half > 0; half >>= 1) {
        if (t < half) {
            s_cs[t] += s_cs[t + half];
            s_hits[t] += s_hits[t + half];
            s_cnt[t] += s_cnt[t + half];
            if (s_ff[t + half] < s_ff[t]) s_ff[t] = s_ff[t + half];
        }
        __syncthreads();
    }
    if (t == 0) out[blockIdx.x] = Partial{s_cs[0], s_hits[0], s_cnt[0], s_ff[0]};
}

class CudaDevice final : public GpuDevice {
public:
    explicit CudaDevice(const CudaOptions& opt) : opt_(opt) {}
    ~CudaDevice() override {
        cudaFree(d_target_);
        cudaFree(d_partials_);
        cudaFree(d_bbp_);
    }

    void init() {
        const auto t0 = std::chrono::steady_clock::now();
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (opt_.device_index < 0 || opt_.device_index >= count) throw GpuError("cuda: no such device");
        check(cudaSetDevice(opt_.device_index), "cudaSetDevice");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, opt_.device_index), "cudaGetDeviceProperties");
        int drv = 0, rt = 0;
        cudaDriverGetVersion(&drv);
        cudaRuntimeGetVersion(&rt);
        info_.api = "cuda";
        info_.name = prop.name;
        info_.vendor = "NVIDIA";
        info_.compute_units = static_cast<uint32_t>(prop.multiProcessorCount);
        info_.driver = "CUDA driver " + std::to_string(drv / 1000) + "." + std::to_string((drv % 1000) / 10) +
                       ", runtime " + std::to_string(rt / 1000) + "." + std::to_string((rt % 1000) / 10) +
                       ", sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
        max_blocks_ = static_cast<uint32_t>((kThreadsTarget + kBlock - 1) / kBlock);
        check(cudaMalloc(&d_partials_, sizeof(Partial) * max_blocks_), "cudaMalloc(partials)");
        h_partials_.resize(max_blocks_);
        // Контекст создаётся лениво: первый вызов ядра не должен попасть в замер.
        check(cudaFree(nullptr), "context init");
        info_.init_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    const GpuInfo& info() const override { return info_; }

    void prepare(const KernelParams& p) override {
        params_ = p;
        dp_.seed = p.seed;
        dp_.n = p.n;
        dp_.len = p.len();
        dp_.mr_base = p.mr_base;
        dp_.mr_step = p.mr_step;
        dp_.bbp = p.bbp;
        const size_t tbytes = std::max<size_t>(4, p.target.size() * sizeof(int32_t));
        if (tbytes > target_bytes_) {
            cudaFree(d_target_);
            d_target_ = nullptr;
            check(cudaMalloc(&d_target_, tbytes), "cudaMalloc(target)");
            target_bytes_ = tbytes;
        }
        if (!p.target.empty())
            check(cudaMemcpy(d_target_, p.target.data(), p.target.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
                  "cudaMemcpy(target)");
        const size_t abytes = std::max<size_t>(8, static_cast<size_t>(p.bbp.positions) * 8);
        if (abytes > bbp_bytes_) {
            cudaFree(d_bbp_);
            d_bbp_ = nullptr;
            check(cudaMalloc(&d_bbp_, abytes), "cudaMalloc(bbp)");
            bbp_bytes_ = abytes;
        }
        reset_bbp_acc();
        prepared_ = true;
    }

    void reset_bbp_acc() override { check(cudaMemset(d_bbp_, 0, bbp_bytes_), "cudaMemset(bbp)"); }

    void read_bbp_acc(std::vector<uint64_t>& acc) override {
        acc.assign(params_.bbp.positions, 0);
        if (!acc.empty())
            check(cudaMemcpy(acc.data(), d_bbp_, acc.size() * 8, cudaMemcpyDeviceToHost), "cudaMemcpy(bbp)");
    }

    RangeResult run_range(uint64_t begin, uint64_t count) override {
        if (!prepared_) throw GpuError("cuda: run_range before prepare");
        RangeResult res;
        if (count == 0) return res;
        // MR: автомату (MrLane) нужно >= 32 чисел на поток (см. vulkan_device.cpp).
        const uint64_t min_steps = params_.kernel == KernelId::MillerRabin ? 32 : 1;
        const uint64_t threads =
            std::max<uint64_t>(1, std::min<uint64_t>((count + min_steps - 1) / min_steps, kThreadsTarget));
        const auto blocks = static_cast<uint32_t>(std::min<uint64_t>((threads + kBlock - 1) / kBlock, max_blocks_));
        const uint64_t total = static_cast<uint64_t>(blocks) * kBlock;
        const uint64_t steps = (count + total - 1) / total;
        auto* acc = reinterpret_cast<unsigned long long*>(d_bbp_);
        switch (params_.kernel) {
        case KernelId::MonkeyRandom:
            range_kernel<0><<<blocks, kBlock>>>(dp_, d_target_, begin, count, steps, d_partials_, acc);
            break;
        case KernelId::MonkeyBrute:
            range_kernel<1><<<blocks, kBlock>>>(dp_, d_target_, begin, count, steps, d_partials_, acc);
            break;
        case KernelId::Bbp:
            range_kernel<2><<<blocks, kBlock>>>(dp_, d_target_, begin, count, steps, d_partials_, acc);
            break;
        case KernelId::MillerRabin:
            range_kernel<3><<<blocks, kBlock>>>(dp_, d_target_, begin, count, steps, d_partials_, acc);
            break;
        }
        check(cudaGetLastError(), "kernel launch");
        check(cudaDeviceSynchronize(), "kernel execution");
        check(cudaMemcpy(h_partials_.data(), d_partials_, sizeof(Partial) * blocks, cudaMemcpyDeviceToHost),
              "cudaMemcpy(partials)");
        for (uint32_t b = 0; b < blocks; ++b) {
            RangeResult part;
            part.checksum = h_partials_[b].checksum;
            part.hits = h_partials_[b].hits;
            part.count = h_partials_[b].count;
            part.first_full = h_partials_[b].first_full;
            res.merge(part);
        }
        if (res.count != count)
            throw GpuError("cuda: kernel processed " + std::to_string(res.count) + " of " +
                           std::to_string(count) + " indices");
        return res;
    }

private:
    CudaOptions opt_;
    GpuInfo info_;
    KernelParams params_;
    DevParams dp_{};
    int32_t* d_target_ = nullptr;
    size_t target_bytes_ = 0;
    Partial* d_partials_ = nullptr;
    std::vector<Partial> h_partials_;
    uint32_t max_blocks_ = 0;
    void* d_bbp_ = nullptr;
    size_t bbp_bytes_ = 0;
    bool prepared_ = false;
};

} // namespace

bool cuda_compiled() { return true; }

std::unique_ptr<GpuDevice> create_cuda_device(const CudaOptions& opt, std::string* err) {
    auto d = std::make_unique<CudaDevice>(opt);
    try {
        d->init();
    } catch (const GpuError& e) {
        if (err) *err = e.what();
        return nullptr;
    }
    return d;
}

std::vector<std::string> list_cuda_devices() {
    std::vector<std::string> out;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return out;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
            out.push_back(std::string(prop.name) + " (" + std::to_string(prop.multiProcessorCount) + " SM)");
    }
    return out;
}

} // namespace monkey
