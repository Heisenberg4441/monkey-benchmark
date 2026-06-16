# Infinite Monkey Benchmark

<p align="center">
  <img src="https://img.shields.io/badge/C++-17-blue.svg" />
  <img src="https://img.shields.io/badge/Architecture-x86__64%20%7C%20ARM64-green.svg" />
  <img src="https://img.shields.io/badge/Accelerator-CUDA-76B900.svg" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" />
</p>

<p align="center">
  <b>English</b> · <a href="README.ru.md">Русский</a>
</p>

A cross-platform compute benchmark built around the Infinite Monkey Theorem. The program either generates random strings or deterministically enumerates the combination space until it matches a reference text, measuring sustained throughput in operations per second (Ops/sec).

The benchmark is designed to evaluate CPU and GPU performance on a task with predictable, tunable computational complexity — parameterized by alphabet size and string length. This makes it straightforward to compare architectures and to observe how throughput scales as the entropy of the input changes.

---

## Table of Contents

- [Theory](#theory)
- [Features](#features)
- [Building](#building)
  - [CPU Backend](#cpu-backend)
  - [CUDA Backend](#cuda-backend)
- [Usage](#usage)
- [Locale Benchmark Methodology](#locale-benchmark-methodology)
- [Architecture](#architecture)
- [Vision & Roadmap](#vision--roadmap)
- [Limitations](#limitations)
- [License](#license)

---

## Theory

According to the Infinite Monkey Theorem, a random sequence of characters generated for an unbounded amount of time will almost surely contain any given finite text.

The probability that a random string of length `L` over an alphabet of cardinality `N` matches the reference is:

$$P = \frac{1}{N^L}$$

The expected number of attempts before the first match is `N^L`. Thus alphabet size and string length define the computational complexity of the task, while the measured quantity is the sustained rate at which candidates are generated and compared per unit of time.

The benchmark captures how throughput varies with alphabet cardinality (locale) and character properties (single- vs. multi-byte), reflecting the memory and arithmetic behavior of the target platform under load.

---

## Features

- **Two operating modes**
  - `random` — Monte Carlo. Each thread independently generates random strings. No shared state between threads and zero coordination overhead.
  - `brute` — deterministic enumeration. The combination space is treated as a number in base `N`; threads split it into disjoint ranges and are guaranteed to cover the entire space without repeats.
- **Minimal hot-loop allocations.** Working buffers are allocated once per thread before the loop; inside the loop memory is only overwritten.
- **UTF-8 support.** Processing happens at the level of Unicode characters rather than bytes: Latin, Cyrillic, CJK ideographs, and emoji are parsed and compared correctly.
- **Dynamic alphabet construction.** The alphabet is derived from the unique characters of the reference file, which makes preparing different-locale scenarios trivial.
- **Heterogeneous backends, any GPU.** A single C++17 core runs on the CPU (x86_64 with AVX2/AVX-512, ARM64 with NEON), on NVIDIA GPUs via **CUDA**, and on any other GPU (AMD / Intel / Apple) via **Vulkan compute**. All backends run the *same* counter-based PRNG — **Philox4x32-10** (Salmon et al., SC'11) — implemented once and shared verbatim across CPU, CUDA and GLSL, so the generated streams are bit-identical (see [Determinism](#determinism)). The GPU API is chosen automatically — NVIDIA → CUDA, otherwise → Vulkan — and can be forced with `--gpu-api`.

---

## Building

The project builds with CMake. The CUDA backend is enabled automatically if a CUDA Toolkit is detected; otherwise only the CPU variant is built.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Resulting binary: `build/monkey_bench`.

### CPU Backend

The CPU variant has no extra dependencies. CMake enables `-O3` and `-march=native` (when the compiler supports it) automatically, so no separate Clang/GCC/MSVC commands are required.

### CUDA Backend

The benchmark supports execution on NVIDIA GPUs via CUDA. Each candidate maps to a single CUDA thread, providing massive parallelism in `random` mode and dense coverage of the combination space in `brute` mode.

Requirements:

- CUDA Toolkit 11.0 or newer
- A GPU with Compute Capability 6.0+
- An NVIDIA driver compatible with the installed Toolkit

If the Toolkit is installed, the same build command picks it up automatically. Target GPU architectures can be overridden when needed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

CUDA can be force-disabled with `-DUSE_CUDA=OFF`.

### Vulkan Backend

For non-NVIDIA GPUs (AMD, Intel, Apple via MoltenVK) the benchmark uses a Vulkan compute backend. CMake enables it automatically when a Vulkan SDK and a GLSL compiler (`glslangValidator` or `glslc`) are found; the GLSL shader is compiled to SPIR-V and embedded into the binary.

Requirements:

- Vulkan SDK (headers + loader) and `glslangValidator`/`glslc` to **build**
- At runtime, only a Vulkan-capable GPU driver (which ships the loader; Windows 10+ also bundles it)

Vulkan can be force-disabled with `-DUSE_VULKAN=OFF`.

The prebuilt release binaries for Windows and Linux already include **both** CUDA and Vulkan, so on any machine with a GPU it just works — no SDK needed.

### Runtime backend selection

`-cpu` / `-gpu` / `-all` choose the load; `--gpu-api auto|cuda|vulkan` chooses the GPU API. The default `auto` picks **CUDA on NVIDIA and Vulkan on any other GPU**, and falls back to the CPU if no GPU is available. On startup the chosen device is printed. Forcing the API lets you compare CUDA vs Vulkan on the same card.

---

## Usage

```bash
./build/monkey_bench [OPTIONS] <reference_file>
```

| Option | Description |
| --- | --- |
| `-m`, `--mode <random\|brute>` | Operating mode. Default: `random`. |
| `-b`, `--backend <cpu\|gpu\|all>` | Target load. Default: `cpu`. |
| `-cpu` / `-gpu` / `-all` | Short aliases for `--backend`. |
| `--gpu-api <auto\|cuda\|vulkan>` | GPU API. Default `auto` (NVIDIA → CUDA, else Vulkan). Force a backend to compare them on the same card. |
| `-t`, `--threads <N>` | Number of CPU threads. Default: all cores. |
| `-d`, `--duration <sec>` | Stop after N seconds (benchmark mode). |
| `--seed <N>` | Seed for the counter-based PRNG (reproducible runs). Default: fixed constant. |
| `--batch-size <N>` | Hot-loop iterations between counter syncs (CPU flush interval; GPU iters/thread per dispatch). Default: 8192. |
| `-h`, `--help` | Argument help. |

The `-all` mode runs the CPU and GPU backends simultaneously and reports combined throughput with a per-device breakdown. Because an exact match is practically unreachable for long strings, use `--duration` for measurements: the program runs for the given time and reports a stable Ops/sec figure.

```bash
# Measure on CPU for 10 seconds
./build/monkey_bench reference.txt -cpu -d 10

# Combined CPU + GPU load
./build/monkey_bench reference.txt -all -d 10

# Deterministic enumeration of a short string on the GPU
./build/monkey_bench reference.txt -m brute -gpu
```

---

## Locale Benchmark Methodology

A key use case is comparing how a platform handles different entropy at a fixed string length. Prepare a set of files of equal length but from different writing systems and compare the Ops/sec.

Example references of 5–10 characters:

| File | Content | Script | Bytes per char |
| --- | --- | --- | --- |
| `en.txt` | `HelloWorld` | Latin | 1 |
| `ru.txt` | `ПриветМир` | Cyrillic | 2 |
| `jp.txt` | `日本語テスト` | Kanji/Kana | 3 |
| `emoji.txt` | `🚀🌍💡🔥💻` | Emoji | 4 |

Run in `random` mode with a fixed measurement window for each file:

```bash
./build/monkey_bench en.txt -m random -d 10
./build/monkey_bench ru.txt -m random -d 10
./build/monkey_bench jp.txt -m random -d 10
./build/monkey_bench emoji.txt -m random -d 10
```

At equal string length, throughput typically drops as alphabet cardinality and per-character byte size grow: the random sampling range widens, and comparing multi-byte characters requires more memory work.

---

## Architecture

### Random Mode

Each thread keeps a local buffer the size of the reference. In a loop:

1. The buffer is filled with alphabet indices drawn from the shared Philox4x32-10 counter-based PRNG (see [Determinism](#determinism)). The candidate's `(seed, key, position)` maps directly to a counter — no per-thread generator state.
2. The candidate is compared character by character against the reference.
3. On a mismatch the buffer is overwritten without new allocations.
4. Each thread keeps its attempt count in a register and periodically (every `--batch-size` iterations) publishes a snapshot to its own cache-line-padded slot. The hot loop performs no atomic read-modify-write on shared memory; the main thread sums the per-thread snapshots to print statistics. See [BENCH.md](BENCH.md) for the contention this avoids.

### Brute Force Mode

The string is treated as a number in base `N`, where `N` is the alphabet cardinality:

1. Threads split the numeric space into disjoint chunks.
2. The starting number of each chunk is converted into a character combination.
3. Enumeration proceeds by carry-propagating increment.
4. This guarantees no duplicated work between threads and full coverage of the space.

### CUDA Backend

On the GPU, work is laid out across a grid of blocks and threads. In `random` mode each thread evaluates the same Philox4x32-10 generator over its own counter range (no cuRAND, no per-thread RNG state) and checks candidates in registers. In `brute` mode the global thread index maps to an offset in the numeric space, giving deterministic coverage. Attempt counters are aggregated via atomic operations and on-device reduction, minimizing global-memory traffic.

### Determinism

The PRNG is the counter-based **Philox4x32-10** (J. Salmon, M. Moraes, R. Dror, D. Shaw, *"Parallel Random Numbers: As Easy as 1, 2, 3"*, SC'11; reference implementation: Random123). It is stateless: each output is a pure function of a 128-bit counter and a 64-bit key, which makes it trivially parallelizable and identical regardless of device.

A single implementation lives in [`src/philox.h`](src/philox.h) (used by CPU and, via `__host__ __device__`, by CUDA) and a line-for-line GLSL port in [`src/philox.glsl`](src/philox.glsl) (shared by the Vulkan shader). The claim that all backends produce identical streams is backed by tests, not assertion:

- `test_philox` — published known-answer vectors (Random123 `kat_vectors`).
- `test_glsl_philox` — the GLSL port compared to the CPU canon over 2¹⁶ outputs.
- `test_vulkan_philox` — the **real compiled SPIR-V** dispatched on a Vulkan device (Mesa lavapipe in CI) vs the CPU canon.
- `test_cuda_philox` — the same check on a CUDA device.

Runs are reproducible: the seed is fixed by default and configurable with `--seed`.

---

## Vision & Roadmap

The long-term goal of this project is to grow from a narrow "monkey" test into an **extensible benchmarking platform for computationally hard search problems**. In the same way that computing the digits of π became a canonical stress test, the aim is to provide a standardized set of "search-solvable" tasks and a unified metric for comparing CPUs and GPUs.

Planned directions:

- **Pluggable workloads.** A "task" abstraction on top of the enumeration/search engine — string matching is just the first workload. Next: numeric search, conjecture checking, hash preimages, combinatorial spaces.
- **π-style numeric tasks.** Workloads with a verifiable reference result (digits of constants, prime/perfect-number search, counterexamples) where a "match" is a found solution.
- **Comparable scoring.** A single performance figure normalized by task difficulty, suitable for cross-machine leaderboards.
- **More backends.** Beyond CUDA: ROCm/HIP, SYCL, and multi-node execution.

---

## Limitations

- In `random` mode, finding an exact match for long or high-entropy strings is practically unreachable: the expected time grows as `N^L`. To verify correctness (that it actually stops on a match), use short strings of 3–5 characters.
- `brute` mode is deterministic, but the `N^L` space quickly exceeds the range of a 64-bit index; very long strings require an extended-precision index.

---

## License

Distributed under the MIT License.
