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
