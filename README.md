# Infinite Monkey Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/C++-17-blue.svg" />
  <img src="https://img.shields.io/badge/Architecture-x86__64%20%7C%20ARM64-green.svg" />
  <img src="https://img.shields.io/badge/GPU-CUDA%20%7C%20Vulkan-76B900.svg" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>

<p align="center">
  <b>English</b> · <a href="README.ru.md">Русский</a>
</p>

A cross-platform CPU and GPU compute benchmark. Run it with no arguments and it
runs a fixed suite of integer workloads on one CPU thread, on all CPU threads and
on every available GPU API (CUDA, Vulkan). Every result is checked against golden
values built into the binary, and the run ends with scores and a JSON report that
can go to a leaderboard. The name comes from the original workload: monkeys
typing random text until they hit a reference string.

```text
$ monkey_bench
monkey_bench 1.0.0 (…) · suite v1 · standard
CPU:  Apple M4 Pro · 14 threads (10P + 4E) · 24.0 GiB · macOS 27.0 arm64
GPU:  Apple M4 Pro · Vulkan (MoltenVK 1.4.2, u64 native) · init 0.04 s
…
Scores — suite v1, standard (3 runs per test, median), 1000 = Apple M4 Pro (10P+4E), macOS 27.0, MoltenVK 1.4.2

Test                      CPU-ST          CPU-MT      GPU Vulkan
monkey-random         1055 ±0.4%      1001 ±0.1%      1000 ±0.1%
monkey-brute          1034 ±0.3%       999 ±0.1%       998 ±0.2%
bbp-pi                1041 ±0.2%       999 ±0.1%       997 ±0.1%
mr-prime64            1001 ±0.0%       999 ±0.2%       999 ±0.1%
Category                    1033            1000             998

CPU score:        1016   geomean(CPU-ST, CPU-MT)
GPU score:         998   best API: GPU Vulkan
Overall score:    1010   geomean(CPU-ST, CPU-MT, GPU)
```

---

## Contents

- [Quick start](#quick-start)
- [What suite v1 measures](#what-suite-v1-measures)
- [Scores and how to read them](#scores-and-how-to-read-them)
- [Methodology](#methodology)
- [Commands](#commands)
- [JSON report](#json-report)
- [Building](#building)
- [The original monkey search (custom mode)](#the-original-monkey-search-custom-mode)
- [Limitations](#limitations)

---

## Quick start

Download the archive for your platform from the releases page and run:

```bash
./monkey_bench                       # standard suite: 3 runs per test, ~4 min
./monkey_bench --quick               # 1 run per test, ~1.5 min (same work, comparable)
./monkey_bench run --json result.json
./monkey_bench verify                # check every kernel on every backend
```

- **Windows / Linux x86_64:** CUDA (NVIDIA) and Vulkan (any GPU) are built in.
- **Linux arm64:** Vulkan.
- **macOS arm64:** Vulkan through MoltenVK, shipped as `libMoltenVK.dylib` next
  to the binary. Unpacked binaries are quarantined by Gatekeeper; run
  `xattr -dr com.apple.quarantine .` in the folder.

Only a normal GPU driver is needed. Without a Vulkan or CUDA driver the binary
still runs the CPU categories and says why the GPU is unavailable.

For a stable result, close heavy applications and plug laptops in.

## What suite v1 measures

Four tests. Each one processes a fixed range of work units `[0, N)`, and every
unit costs about the same. The score is computed from the time this fixed work
takes. N differs per category so that each test runs 3–10 s on the reference
machine.

| Test | Work unit | What it stresses | Result checked against |
| --- | --- | --- | --- |
| `monkey-random` | one random 43-character candidate for *“The quick brown fox jumps over the lazy dog”* (alphabet 28) drawn from Philox4x32-10; result = length of the matched prefix | 32-bit multiply/xor throughput, short data-dependent loops | golden checksum |
| `monkey-brute` | the index written in base 28 and padded to 43 digits; result = length of the matched suffix | 64-bit integer division throughput | golden checksum |
| `bbp-pi` | a block of 256 terms of the Bailey–Borwein–Plouffe series for hex digits of π near position 10⁶ | 32-bit Montgomery modular exponentiation, long division, instruction-level parallelism | 8 hex digits of every position vs the embedded `reference/pi_hex.txt`, plus a golden checksum |
| `mr-prime64` | deterministic Miller–Rabin test of one odd number above 2⁶² | 64-bit Montgomery multiplication (`mulhi`), branchy control flow (SIMD divergence on GPUs) | golden prime count + checksum, cross-checked with a segmented sieve |

All arithmetic is integer, and it is defined so that a given work unit has
exactly one bit-exact result on every CPU and GPU. There is no floating point:
FP64 results differ between devices, and Apple GPUs have no FP64 at all.

`monkey_bench list` prints the exact work per category and the reference times.

## Scores and how to read them

- **Test score** = `1000 × t_ref / t`. Here `t` is the median time of the runs
  and `t_ref` is the time on the reference machine (Apple M4 Pro, 10P + 4E,
  macOS 27.0, its GPU through MoltenVK 1.4.2). A score of 2000 means the test
  ran twice as fast as on the reference machine.
- **Category score** = geometric mean of its four tests. The categories are
  CPU-ST (one thread), CPU-MT (all logical threads) and GPU, scored once per
  available API (GPU CUDA, GPU Vulkan). Every API uses the same work and the
  same `t_ref`.
- **CPU score** = geomean(CPU-ST, CPU-MT). **GPU score** = the best GPU API.
- **Overall score** = geomean(CPU-ST, CPU-MT, GPU). It is not computed on a
  machine without a GPU: there it would not be comparable. The report marks
  that run `partial`.
- **±x %** next to a score is the coefficient of variation of the run times.
  Above 3 % the test is marked `unstable` (`*`). The score still counts, but
  the result is noisy: thermal throttling, background load, power saving.
- **INVALID** means a checksum, a prime count or a π digit did not match the
  golden value. That test gets no score, and neither do its category and the
  overall score.
- **Comparability.** Scores are comparable only within the same
  `suite_version`. Any change to a kernel, a work size or a golden value bumps
  it. The report also marks itself not comparable for custom runs, stress runs,
  debug builds and `-march=native` builds.

Geometric means keep one exceptional test from dominating a category. Doubling
one of four tests raises the category by 19 %, not by 25 %.

## Methodology

**Fixed work, not "how much in N seconds".** Each test has a fixed number of
work units, and the clock measures how long they take. An older version
counted operations per second for a set duration. There the cost of an
operation grew with the counter (Miller–Rabin, BBP), so the result depended on
the duration and the thread count.

**One session per device.** CPU threads and GPU contexts are created once. For
every test and category, the engine then runs:

1. **Warmup**, 2 s (1 s in `--quick`). Clocks ramp up and the chunk size is
   tuned to ~15 ms per CPU chunk and ~30 ms per GPU dispatch. That is far from
   the 2 s Windows TDR limit, and a stop takes effect within one chunk.
2. **Measured runs**, 3 in `--standard` and 1 in `--quick`. All workers start
   on a common barrier. Time runs on `steady_clock` until the last chunk
   finishes. Device initialization is reported separately (`init_time_s`) and
   is never part of a measurement.

**Shared chunk dispatcher.** Work is handed out in chunks by a single atomic
compare-and-swap per chunk, not one per operation. The CPU threads and the GPU
share one dispatcher in `--all`: nothing is computed twice, and faster devices
simply take more chunks.

**Order-independent checksums.** The result of unit `i` enters
`Σ fmix64(fmix64(i) ⊕ r) mod 2⁶⁴` (splitmix64 finalizer; units with an empty
result `r = 0` contribute nothing). The sum does not depend on thread count,
chunk size or execution order, so the CPU threads and GPU dispatches can add
up their parts. The processed count and the number of non-empty results are
checked as well.

**Same operations on every backend.** Each workload is one header-only kernel
(`src/workloads/*_kernel.h`, `__host__ __device__` for CUDA) with a
line-by-line GLSL port (`src/shaders/`). The GLSL is compiled in two variants:
native `uint64_t` (`shaderInt64`) and a `uvec2` emulation for devices without
it. `test_parity` and `monkey_bench verify` check that CPU, CUDA and both Vulkan
variants return identical results for fixed ranges.

**Hybrid CPUs.** On macOS benchmark threads run at the highest QoS, so CPU-ST
stays on a performance core. On Linux and Windows threads are not pinned and
the OS scheduler decides. The report records the P/E core counts.

**Stress mode.** `run --stress -d 60` repeats each test's fixed window for 60 s
per test and category and verifies every completed pass. It reports throughput
over time and `throttle_ratio`, the throughput of the last 10 % of the time
over the first 10 %. It gives no scores; use it for thermal and stability
analysis.

**Run-to-run stability** of the overall score on the reference machine is
documented in [BENCH.md](BENCH.md).

## Commands

```text
monkey_bench [run] [options]          run the suite and print scores
monkey_bench list                     list the suite tests
monkey_bench info                     system, build and compute devices
monkey_bench verify [options]         check every kernel on every backend
monkey_bench custom [file] [options]  custom workload: ops/s only, not comparable
monkey_bench <file> [options]         legacy syntax, same as custom
```

| `run` option | Meaning |
| --- | --- |
| `--standard` / `--quick` / `--stress` | 3 runs per test (default), 1 run, or timed stress |
| `--cpu` / `--gpu` / `--all` | CPU categories, GPU categories, or both plus an informational CPU+GPU row |
| `--gpu-api auto\|cuda\|vulkan` | GPU APIs to test (default: every available one) |
| `--test NAME` | run only this test (repeatable); category scores need all four |
| `-d, --duration SEC` | stress: seconds per test and category (default 30) |
| `--json FILE` | write the JSON v2 report (`-` for stdout) |
| `--vk-allow-cpu` | allow software Vulkan devices such as Mesa lavapipe |

Exit codes: `0` success, `1` usage or backend error (the run stops with a
message instead of hanging), `2` a result did not match its golden value.

## JSON report

`--json FILE` writes a report that follows
[`bench_output.schema.json`](bench_output.schema.json) (schema v2). CI validates
it with `check-jsonschema`. It contains:

- `schema_version`, `suite_version`, `mode`, `timestamp`;
- `binary`: version, git sha, compiler, flags, build type;
- `system`: OS, kernel, CPU model, physical/logical/performance/efficiency
  cores, RAM;
- `devices`: API, name, vendor, driver, compute units, 64-bit mode, init time;
- `tests[]`: run times, median/min/max/CV, ops/s, checksum and expected
  checksum, validity, score, chunk size, work split between CPU and GPU;
- `scores`: categories, CPU, GPU, overall;
- `flags`: `comparable`, `custom`, `debug_build`, `native_arch`,
  `portable_arith`, `invalid`, `unstable`, `partial`.

Checksums are hex strings: JSON numbers lose precision above 2⁵³.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

- **Vulkan** is built in when the Vulkan headers and `glslangValidator` are
  found (Vulkan SDK; on Linux `libvulkan-dev glslang-tools`; on macOS
  `brew install vulkan-headers vulkan-loader glslang molten-vk`). The Vulkan
  loader is opened at run time, so the binary does not depend on it.
- **CUDA** is built in when a CUDA Toolkit (11+) is found; the runtime is
  linked statically.

| CMake option | Default | Meaning |
| --- | --- | --- |
| `USE_CUDA` / `USE_VULKAN` | `ON` | enable the backend if its SDK is found |
| `MONKEY_NATIVE_ARCH` | `OFF` | `-march=native`; such builds are flagged as not comparable |
| `MONKEY_PORTABLE_ARITH` | `OFF` | force the portable 32-bit paths of the kernels (what MSVC and GLSL use); CI runs the tests this way too |
| `MONKEY_BUILD_TESTS` | `ON` | Catch2 unit tests, backend parity tests and CLI smoke tests |

`tools/suite_gen.cpp` (`cmake --build build --target monkey_suite_gen`)
regenerates the golden values of the suite.

CI builds and tests on Linux x86_64 and arm64 (Vulkan via Mesa lavapipe),
Windows x86_64 (MSVC), macOS arm64 (MoltenVK) and a build with the portable
arithmetic. It also compiles the CUDA backend.

## The original monkey search (custom mode)

The infinite monkey theorem: a random sequence of characters typed forever
almost surely contains any given text. A random string of length `L` over an
alphabet of `N` characters matches the reference with probability `1/N^L`.
Custom mode runs that search on your own reference file until an exact match is
found or the time runs out:

```bash
./monkey_bench custom reference.txt -m random -d 10   # random candidates, 10 s
./monkey_bench custom reference.txt -m brute -gpu     # enumerate the space on the GPU
./monkey_bench reference.txt -all -d 10               # legacy syntax: CPU + GPU together
./monkey_bench custom -w bbp -d 10                    # suite kernels, timed
```

The alphabet is the set of distinct UTF-8 characters of the reference, and the
hot path compares alphabet indices. Throughput therefore depends only on the
alphabet size and the string length. The script (Latin, Cyrillic, CJK, emoji)
and the bytes per character do not matter. Brute force uses a 64-bit index:
references whose code exceeds 2⁶⁴ cannot be found by enumeration, and the
program says so. Custom runs report ops/s from completed chunks and are never
comparable with suite scores.

## Limitations

- **Integer compute only.** The suite exercises integer ALU, multiply, divide
  and branch throughput. It does not measure memory bandwidth, caches, floating
  point or I/O. It is not a general-purpose replacement for Geekbench or
  SPEC.
- **One reference machine.** Scores are relative to an Apple M4 Pro. Its GPU
  runs through MoltenVK (Vulkan on Metal); native Metal is out of scope, and
  MoltenVK overhead may undervalue Apple GPUs a little.
- **Fixed GPU work.** GPU tests are sized for ~5 s on the reference GPU. A slow
  integrated GPU takes proportionally longer; software Vulkan (lavapipe) would
  take hours, so CI checks GPUs with `verify` instead of a full run.
- **CUDA is verified in CI by compilation only.** No NVIDIA GPU was available
  while developing suite v1. On an NVIDIA machine run `monkey_bench verify`
  once: it compares the CUDA results with the golden values.
- **Microarchitecture sensitivity is intended.** For example, `monkey-brute`
  depends on 64-bit division throughput, which differs several-fold between
  CPU cores and is emulated on GPUs.
- **Laptops and hybrid CPUs.** Thermal limits and scheduling on hybrid CPUs
  outside macOS add noise. Watch the `unstable` flag, and use `--stress` to see
  throttling.

## License

Distributed under the MIT License.
