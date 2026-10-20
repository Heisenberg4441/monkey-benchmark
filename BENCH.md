# Benchmark Notes

Measurements backing the refactoring phases. Numbers are machine-specific;
the point is the *relative* before/after on identical hardware, not absolute
Ops/sec (see the README "Methodology" notes).

## Phase 2 — Removing atomic contention from the hot loop

**Goal.** Each CPU worker keeps its attempt counter in a register and flushes a
snapshot to its *own* cache-line-padded `std::atomic` slot only every
`--batch-size` iterations. The reporter sums the per-thread snapshots. No
atomic read-modify-write on shared memory happens in the hot loop. On CUDA the
per-thread `atomicAdd` was replaced by a shared-memory block reduction with a
single global `atomicAdd` per block.

### Test machine

- Apple Silicon, 14 logical cores (`hw.ncpu = 14`), macOS.
- `-DNATIVE_ARCH=OFF -DCMAKE_BUILD_TYPE=Release`, Vulkan via MoltenVK.
- Workload: `random` mode, reference = `"The quick brown fox jumps over the lazy
  dog"` (len 43, alphabet 27), `-t 14 -d 3`, 3 runs each.

### End-to-end throughput (random, 14 threads, default batch 8192)

| Build | Ops/sec (median of 3) |
| --- | --- |
| Before (shared atomic, batched at 8192) | ~1.538 G |
| After (per-thread slots, batch 8192)    | ~1.538 G |

**No change at the default batch — and that is expected.** The pre-Phase-2 code
already amortized the shared atomic by flushing only every 8192 iterations, so
contention was already negligible *at that batch*. The end-to-end win the
original review anticipated (+20–60%) does not materialize here precisely
because the baseline was already batched; reporting it as a speedup would be
dishonest on this configuration.

### Why the change still matters: batch-size sensitivity

The per-thread design makes counter-sync frequency essentially free, so a small
batch (responsive reporting, fine-grained future workloads) no longer costs
throughput:

| `--batch-size` | Ops/sec (after) |
| --- | --- |
| 1     | ~1.518 G |
| 256   | ~1.533 G |
| 8192  | ~1.537 G |
| 65536 | ~1.535 G |

Flushing *every iteration* costs ~1% versus the default.

### The contention being removed (isolated microbenchmark)

To show the effect the batching was hiding, a focused microbenchmark counts
`ITERS` increments per thread two ways: (a) `fetch_add` to one shared atomic
every iteration, (b) a relaxed store to a per-thread padded slot every
iteration.

| Threads | (a) shared atomic / iter | (b) per-thread slot / iter | speedup |
| --- | --- | --- | --- |
| 1  | 0.58 Gops | 4.33 Gops  | ×7.5 |
| 4  | 0.12 Gops | 14.52 Gops | ×125 |
| 14 | 0.03 Gops | 29.73 Gops | ×1189 |

A single shared atomic collapses under cross-core traffic (0.03 Gops at 14
threads); per-thread slots scale linearly. This is the contention the new design
eliminates structurally — the hot loop now *never* touches shared memory, so the
result holds at any thread count and any batch size, not just the one batch
value that happened to be tuned in.

### Correctness

`tests/test_counters.cpp` proves the snapshot mechanism is lossless: the summed
per-thread counters equal the exact number of iterations (single-threaded,
8-thread concurrent stress, and a mid-run monotonic/bounded snapshot check).
This is the deterministic form of the acceptance criterion "final counter sum ==
number of iterations".

### GPU (CUDA)

Not measured here (no NVIDIA GPU on the test machine). The change — shared-memory
block reduction with one global `atomicAdd` per block instead of one per thread —
will be measured on a CUDA device in Phase 6 CI / on hardware and appended here.

## Phase 3 — Pluggable workload abstraction

Same machine as Phase 2. The CPU backend was refactored from a hardcoded
`random_worker`/`brute_worker` to a generic worker driving an `IWorkload`
(`execute_batch(counter_start, batch_size, result)`). The monkey logic moved
into `MonkeyWorkload`, now fully counter-based (each global counter maps to one
candidate via Philox; threads take disjoint counter ranges).

### Monkey throughput: before (Phase 2) vs after (Phase 3 abstraction)

Single thread, `random`, `"The quick brown fox…"`, `-d 2`, back-to-back,
CPU-only builds:

| Build | Ops/sec |
| --- | --- |
| Phase 2 (`random_worker`, free function)     | ~151 M |
| Phase 3 (`MonkeyWorkload::execute_batch`)    | ~127 M |

**Honest finding: a ~16% regression on monkey-`random`, and only there.**
- `brute` mode is unaffected (~1.34 M both) — same loop structure, no PRNG.
- The Philox math is byte-identical between the two (verified: with matched
  counter packing and a single thread, both call `philox4x32_10` with the same
  inputs), so this is **not** an algorithmic change — it is instruction
  scheduling / register allocation around the uint64-counter virtual-workload
  path under Apple clang on arm64. The regression reproduces single-threaded,
  so it is per-candidate codegen, not contention or threading.

Things tried that did **not** recover it (ruling out the obvious causes):
accumulating `iterations` in a register instead of through the `out&` reference;
manually inlining the random path to drop the `monkey_match` call and `mode`
branch; matching the Phase 2 Philox counter layout exactly (`{cand,pos,0,0}`).
All stayed at ~127 M.

**Decision.** Accepted as the cost of the abstraction on this microarchitecture.
The pluggable interface is the deliverable of Phase 3 and the enabler for the
heavy compute workloads (BBP, Miller-Rabin) where per-op cost dwarfs this delta;
`brute` (the search the project is named for) is unchanged. The delta is
arm64/Apple-clang-specific and untested on x86 — Phase 6 CI re-measures on
`x86-64-v3` and the result will be appended here before claiming parity.

---

# Suite v1 conversion (2026-10)

Machine for every measurement below unless noted: Apple M4 Pro (10P + 4E,
14 threads), macOS 27.0, Apple clang 21, Release, no `-march=native`.
Baseline is master `61bc279`; the audit numbers are in `docs/ROADMAP.md` §0.4.

## Miller-Rabin: Montgomery kernel

The kernel moved from `(a·b) % n` on `unsigned __int128` (GCC/Clang) or a
bit-serial add-and-double loop (MSVC/CUDA) to Montgomery REDC with R = 2^64
(`workloads/arith.h`). Witness 2 uses left-to-right exponentiation where
multiplying by the base is a modular doubling; R^2 mod n is only computed for
numbers that survive witness 2.

Kernel microbenchmark, 1 thread, 2^20 consecutive odd n above 2^k, best of 5,
same prime counts for old and new:

| n ≈ | old, native `__int128 %` | new, native `mulhi64` | speedup |
| --- | --- | --- | --- |
| 2^20 | 319 ns | 110 ns | ×2.9 |
| 2^32 | 543 ns | 151 ns | ×3.6 |
| 2^48 | 759 ns | 211 ns | ×3.6 |
| 2^62 | 655 ns | 264 ns | ×2.5 |

The path MSVC and CUDA used to take (no `__int128`), emulated on clang with
`-U__SIZEOF_INT128__ -DMONKEY_PORTABLE_ARITH`:

| n ≈ | old, bit-serial mulmod | new, 32-bit-limb `mulhi64` | speedup |
| --- | --- | --- | --- |
| 2^20 | 1028 ns | 223 ns | ×4.6 |
| 2^62 | 5788 ns | 384 ns | ×15.1 |

The bit-serial loop costs grow with the bit length of n, so the "×4.5" quoted
in the audit holds only for small n. It was also wrong for n ≥ 2^63
(97 % wrong products in the audit probe).

End to end (old timed engine, `-w miller-rabin -t 1`): 6.62 → 16.6 M ops/s at
`-d 1`, 5.75 → 15.9 M ops/s at `-d 6`. The drift with duration is still there,
because this engine still walks n upward from 0. It disappears with the fixed
2^62 window in suite v1.

## BBP: integer fixed-point kernel

`double` series + 128-bit `mulmod64` replaced by a uint64 fixed-point sum
(ulp 2^-64), 32-bit Montgomery powers of 16 for four moduli in lockstep (ILP),
and two-limb long division for each term. No FP64 — Apple GPUs have none
(`shaderFloat64 = false` on MoltenVK).

One full position, 1 thread:

| Kernel | d | s / position |
| --- | --- | --- |
| old, `double` + `__int128 %` | 999 984 – 999 999 | 0.674 |
| new, Montgomery, ×16 as a Montgomery multiply (shipped) | 999 000 | 0.098 |
| new, ×16 as four modular doublings | 999 000 | 0.106 |
| new, hardware `%` instead of Montgomery | 999 000 | 0.093 |

All three new variants produce the identical 64-bit sum. Hardware `%` is 5 %
faster on M4 Pro, but 64-bit `div` is much slower on many x86 cores and is
emulated on GPUs, so the Montgomery variant ships as the single kernel for
all backends. The `MONKEY_PORTABLE_ARITH` build runs at the same speed
(0.098 s): this kernel only needs 32×32→64 products.

Accuracy: 8 hex digits per position match `reference/pi_hex.txt` for
positions 0–2047 and at both ends of the suite window near 10^6 (`test_bbp`).
The digits are now embedded in the binary, so BBP no longer depends on the
current directory.

## Monkey: canonical counter → candidate mapping

`IWorkload` and the 128-bit `Counter` are replaced by `KernelParams` +
`cpu_run_range()` (`src/kernels.h`). The CPU's Philox mapping
(`ctr = {pos, i_lo, i_hi, 0}`, `key = {seed, 0}`) is now the canon that every
backend implements. The per-index result is the matched prefix (random) or
suffix (brute) length, and it goes into the order-independent checksum
together with `count` and `hits`. Brute is a 64-bit index in base n padded
to `len` digits, so the fox string (n = 28, len = 43, 28^43 > 2^128) can be
enumerated now; before, it failed `verify()`.

Old timed engine, fox string, `-d 4`, same machine:

| Mode | before (`61bc279`) | after |
| --- | --- | --- |
| random, 1 thread | 131 M/s | 159 M/s |
| random, 14 threads | 1.28 G/s | 1.52 G/s |
| brute, 1 thread | did not run | 1.78 G/s |
| brute, 14 threads | did not run | 16.6 G/s |

The random gain (+21 %) recovers the Phase-3 regression documented above
(151 → 127 M/s when the virtual `IWorkload` path was introduced). The kernel
is now a plain loop over `monkey_random_prefix`, and the call is no longer
virtual. Brute costs ~1.04 64-bit divisions per candidate. M4 Pro pipelines
independent `udiv`s, so it runs at ~0.56 ns per candidate on one core.

## Vulkan: one shader for all kernels, two 64-bit variants

`src/shaders/kernels.comp` ports all four kernels line by line; the kernel is
a specialization constant. 64-bit arithmetic goes through `u64.glsl`, which
is compiled twice: `uint64_t` (`shaderInt64`) and `uvec2` emulation. 64/32
division is Hacker's Delight `divlu` on 32-bit ops in both variants.
`test_parity` checks both variants against `cpu_run_range` on MoltenVK (all
11 cases, 3 different dispatch splits each).

Sustained throughput, M4 Pro via MoltenVK 1.4.2, ~25–120 ms dispatches, 3 s:

| Kernel | u64 native | u64 emulated |
| --- | --- | --- |
| monkey-random (fox) | 4.83 G/s | 4.69 G/s |
| monkey-brute (fox) | 10.4 G/s | 9.50 G/s |
| bbp (blocks of 1024 terms near 10^6) | 0.447 M/s | 0.448 M/s |
| miller-rabin (odd n above 2^62) | 39.7 M/s | 33.2 M/s |

Device init (instance, device, all four pipelines): 0.36 s native, 0.07 s
emulated (second run, warm pipeline cache). Init is now excluded from the
measurement window: the device is created once and reused for the warmup.

Honest notes:
- monkey-random was 5.15 G/s on the old shader at `--batch-size 128`. The
  new one is 6 % slower. It now computes the canonical candidate from a
  64-bit index and adds every hit to the checksum. The old shader computed
  different candidates (per-thread seeds) and checked nothing.
- miller-rabin on the GPU is no faster than 14 CPU threads. About 78 % of
  32-lane SIMD groups contain at least one prime, and a prime runs all 7
  witnesses, so lanes that drew composites sit idle. A per-lane state
  machine that keeps all lanes busy is the next optimization (before the
  t_ref calibration, so scores stay comparable).
