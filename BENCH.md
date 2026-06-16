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
