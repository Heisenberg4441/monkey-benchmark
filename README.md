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
- **Heterogeneous backends.** A single C++17 core for CPU (x86_64 with AVX2/AVX-512, ARM64 with NEON) and a CUDA backend for massively parallel execution on NVIDIA GPUs.

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

The backend is selected at runtime with `--backend` (or the `-cpu` / `-gpu` / `-all` aliases). On startup the benchmark prints the selected device and the number of SMs in use. If the CUDA backend is unavailable on the target system, the program automatically falls back to the CPU.

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
| `-t`, `--threads <N>` | Number of CPU threads. Default: all cores. |
| `-d`, `--duration <sec>` | Stop after N seconds (benchmark mode). |
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

1. The buffer is filled with random alphabet indices using `std::mt19937`.
2. The candidate is compared character by character against the reference.
3. On a mismatch the buffer is overwritten without new allocations.
4. A global attempt counter (`std::atomic`) is updated in batches to reduce contention; the main thread reads it to print statistics.

### Brute Force Mode

The string is treated as a number in base `N`, where `N` is the alphabet cardinality:

1. Threads split the numeric space into disjoint chunks.
2. The starting number of each chunk is converted into a character combination.
3. Enumeration proceeds by carry-propagating increment.
4. This guarantees no duplicated work between threads and full coverage of the space.

### CUDA Backend

On the GPU, work is laid out across a grid of blocks and threads. In `random` mode each thread initializes an independent generator (cuRAND) and checks candidates in its own register-resident buffer. In `brute` mode the global thread index maps to an offset in the numeric space, giving deterministic coverage. Attempt counters are aggregated via atomic operations and on-device reduction, minimizing global-memory traffic.

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
