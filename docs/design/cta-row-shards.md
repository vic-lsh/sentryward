# Checking a large grid by CTA row shards (k_stage_in/out on H100)

Status: design, for review before code. Owner: Coral SentryWard structural-mode lane.

## Problem

Coral's `k_stage_in` and `k_stage_out` on H100 launch a 512 x 48 grid (24,576 CTAs of 256
threads). Grid row `y` copies 2 MiB between staging row `y` and the state entry
`tab[(req + y) * stride]`. SentryWard executes every CTA (the frontend's per-CTA execution,
`AddCta`, `CtaFacts::PropertyDecisionsSnapshot`) before class admission, and the run exceeds the
16 GiB memory cap at about 290 s (profile, 1 input). The GB10 launch (512 x 8) passes in 293 s.
Class admission does not help: the base of row `y` is a different loaded word, so rows never
join one class, and members are executed before they are admitted.

## Options

A. Cheaper admission (admit a CTA as a translate of its representative without executing it):
   needs a proof that the CTA's control and address shape are the representative's, shifted by a
   known delta, before execution. General but deep (it changes the frontend's unit-major loop).

B. CTA row shards (recommended): check the launch as `K` inputs, shard `k` executing only the
   CTAs with `blockIdx.y` in `[y_k, y_{k+1})`, with the launch's true grid and words (no
   renumbering: addresses are exactly the launch's). Each shard checks containment and race
   freedom among its own CTAs. Cross-shard race freedom is a separate exact check over the
   declared write pieces, plus, where a write address is data-dependent, a premise that is
   checked elsewhere.

## B in detail (revision 2, after review 1)

Admitted kernels only. A launch may be sharded only when its PTX has no inter-CTA communication
SentryWard cannot see within a shard: no cluster launch, distributed shared memory, cluster or
grid barrier, cooperative launch, atomics or fences of gpu/sys scope on global memory, and no
global read of a location some CTA of the launch writes unless the reader and the writer are in
the same shard. The first conditions are a static PTX scan (the existing precondition scan,
`unit_classes.cpp` PreconditionScan, extended); the last is part of the cross-shard check below.
A kernel that fails any condition is not sharded.

- Fork input field `cta_rows: { y: [lo, hi) }`: the frontend plans units with the true grid,
  words and cluster shape and executes only the CTAs in the range. Facts that depend on the whole
  grid (class planning, grid extents) are computed from the true grid; the shard's report records
  the full launch identity (kernel digest, grid, block, words, allocations, premises, config).
- Each shard reports its actual accessed byte pieces per allocation (reads and writes, from its
  verified footprint evidence, not from declarations) and its race and containment verdicts.
- Cross-shard check, exact, in `check_verdicts.py` (or a small tool it calls): for every pair of
  shards and every allocation pair that may alias (same identity, or a shared alias scope, or an
  unknown identity: refuse), no write piece of one shard intersects a write piece or a read piece
  of the other. Pieces are interpreted with their base, rows, stride, period, count and width. A
  symbolic piece (a data-dependent base) cannot be compared exactly and refuses the shard set.
- Coverage authentication: the shards of one launch must have identical launch identity, and
  their row ranges must partition `[0, grid.y)` exactly (no gap, no overlap), each covering all x
  and z CTAs; every existing record obligation (case trees, split fields, field reads) holds per
  shard. A row is `checked` only then.

Consequence for Coral today. `k_stage_in`: writes only staging rows `y` (affine, exact pieces),
reads `state` and `tab` (never written by the launch), so its cross-shard check is exact; it can be
sharded. `k_stage_out`: writes `state` at the data-dependent entry `tab[(req + y) * stride]`, a
symbolic piece, so the shard set is refused and the row stays trusted. The verified monitor's
`injective_spec` is not usable as stated: it quantifies over one checked step's tokens and
excludes null outputs, and nothing binds `tab[(req + y) * stride]` of a launch row to a distinct
non-null token output (prefill segments, padding and pad sinks included). A proved row-to-token
binding would be a separate item; this design does not depend on it.

## Soundness obligations (for the reviewer)

1. A shard execution equals the launch restricted to its CTAs, for admitted kernels (no
   inter-CTA communication other than through memory compared exactly across shards).
2. Launch race freedom = within-shard race freedom + no cross-shard write/write or write/read
   byte intersection over possibly aliasing allocations, decided exactly or refused.
3. Launch containment = containment of every shard's accesses.
4. The shard set is an exact partition of one launch with one identity.

## Controls

- `out[y]` written and `out[y + 1]` read by row `y`: the cross-shard write/read check refuses.
- Input and output allocations sharing an alias scope: refused.
- A kernel with a grid barrier or a gpu-scope atomic on global memory: not sharded.
- A shard set with a gap or an overlap in y, or differing launch words: not accepted.
- `k_stage_out`: refused (symbolic write piece), stays trusted.

## Cost

A frontend filter with an extended precondition scan, a generator split, per-shard piece
export and an exact cross-shard check. Measured cost is part of acceptance: for H100
`k_stage_in` (6 shards of 8 rows; one 8-row run is 293 s and about 14 GB on GB10) record the total
and peak per shard; if the total exceeds the 10-minute check budget, shards run as separate inputs
of the record (each under the 900 s box) and the record's wall time is reported, not hidden.
Option A stays open for later.
