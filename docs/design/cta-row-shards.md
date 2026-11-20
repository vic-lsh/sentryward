# Checking a large grid by CTA row shards (k_stage_in/out on H100)

Status: design revision 3 (after review 2), for review before code. Owner: Coral SentryWard structural-mode lane.

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

## B in detail

Admitted kernels only. A shard's execution must equal the launch restricted to its CTAs, so a
launch may be sharded only when its PTX passes an **allowlist** scan (the existing precondition
scan, `unit_classes.cpp` PreconditionScan, extended); any instruction or operand outside the list
refuses sharding:

- Special registers: only `%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`, `%lanemask_*`, and
  `%dynamic_smem_size`. Placement- or time-dependent registers (`%smid`, `%nsmid`, `%warpid`,
  `%nwarpid`, `%clock`, `%clock64`, `%globaltimer*`, `%gridid`, `%cluster*`, `%pm*`) refuse.
- Memory: `ld`/`st` of global, shared, local and param space with weak or relaxed-CTA semantics,
  and generic-space accesses the frontend resolves to one of these; CTA-scope fences and
  barriers. Refused: atomics and reductions of any scope on global memory, gpu/sys-scope fences,
  volatile or `.mmio` accesses, `cp.async.bulk`/TMA and tensor-map operands, `st.async`,
  `multimem`, `mbarrier` of cluster scope, distributed shared memory, cluster and grid barriers,
  cooperative launches, and indirect calls or branches (`call` through a register, `brx.idx`).
  (`cp.async` to shared memory within the CTA is allowed: its source is a global read like `ld`.)
- No global read of a location some CTA of the launch writes unless reader and writer are in the
  same shard; this is part of the cross-shard check below.

- Fork input field `cta_rows: { y: [lo, hi) }`: the frontend plans units with the true grid,
  words and cluster shape and executes only the CTAs in the range. Facts that depend on the whole
  grid (class planning, grid extents) are computed from the true grid; the shard's report records
  the full launch identity (kernel digest, grid, block, words, allocations, premises, config).
- Each shard reports its actual accessed byte pieces per allocation (reads and writes, from its
  verified footprint evidence, not from declarations) and its race and containment verdicts.
- Common coordinates. Every piece is translated to absolute device byte addresses: the launch
  words give each pointer argument's value and the input gives each allocation's base and size,
  so piece `(arg, offset)` becomes the interval `[ptr(arg) + offset, + width)`. Pieces of
  different arguments are compared by these absolute intervals, so a pointer `B` 100 bytes into
  `A` intersects `A`'s byte 100. A piece whose pointer value or allocation is not resolved to
  concrete words refuses the shard set.
- Cross-shard check, exact, in `check_verdicts.py` (or a small tool it calls): for every pair of
  shards, no write piece of one shard intersects a write piece or a read piece of the other, in
  absolute coordinates. Pieces are interpreted with their base, rows, stride, period, count and
  width.
  - A symbolic **write** piece (data-dependent base) refuses the shard set.
  - A symbolic **read** piece is over-approximated by the whole interval of the allocation it is
    contained in (its containment is checked within the shard). It is accepted only if no write
    piece of any shard intersects that whole interval; otherwise the shard set is refused.
- Authentication against the requested launch. The generator writes the expected launch identity
  of the case (kernel digest, grid including `grid.y`, block, words, allocations, premises,
  config) into each shard input. Every shard report must carry exactly that identity, and the
  shards' row ranges must partition `[0, expected grid.y)` exactly (no gap, no overlap), each
  covering all x and z CTAs. A consistent shard set for a different grid (for example
  `grid.y = 1` instead of 48) does not match the expected identity and is refused. Every existing
  per-case record obligation (case trees, split fields, field reads) holds per shard against that
  expected identity. A row is `checked` only then.

Consequence for Coral today. `k_stage_in` writes only staging rows `y` (affine, exact pieces) and
reads `tab` (exact) and `state` at the table-selected entry (a symbolic read piece). The launch
writes neither `tab` nor `state`, and the staging allocation's interval is disjoint from the
`state` allocation's interval, so the symbolic read passes the whole-allocation rule; the shard set
is decided exactly and the row can be checked. If the staging and state intervals were not
resolved, or overlapped, the set is refused. `k_stage_out` writes `state` at the data-dependent
entry `tab[(req + y) * stride]`, a symbolic write piece, so the shard set is refused and the row
stays trusted. The verified monitor's `injective_spec` does not discharge this: it quantifies over
one checked step's tokens and excludes null outputs, and nothing binds a launch row's entry to a
distinct non-null token output (prefill segments, padding and pad sinks included). A proved
row-to-entry binding would be a separate item; this design does not depend on it.

## Soundness obligations (for the reviewer)

1. A shard execution equals the launch restricted to its CTAs, for kernels passing the allowlist
   (no placement- or time-dependent input, no inter-CTA communication other than plain memory
   compared exactly across shards).
2. Launch race freedom = within-shard race freedom + no cross-shard write/write or write/read
   byte intersection in absolute device addresses, decided exactly (symbolic reads by their whole
   allocation interval) or refused.
3. Launch containment = containment of every shard's accesses.
4. The shard set is an exact partition of the expected launch of the case, with that identity.

## Controls

- `out[y]` written and `out[y + 1]` read by row `y`: the cross-shard write/read check refuses.
- Pointer argument `B` 100 bytes into allocation `A`, row 0 writes `A[100]`, row 47 reads `B[0]`:
  the absolute-interval check refuses.
- A CTA branching on `%smid` (or reading `%clock`): not sharded.
- A global `atom` of any scope, or a `cp.async.bulk` with a tensor map: not sharded.
- A symbolic read whose allocation interval meets another shard's write: refused.
- A kernel with a grid barrier or a gpu-scope atomic on global memory: not sharded.
- A shard set with a gap or an overlap in y, or differing launch words: not accepted.
- A consistent shard set covering `grid.y = 1` submitted for the expected `grid.y = 48`: refused.
- `k_stage_out`: refused (symbolic write piece), stays trusted.

## Cost

A frontend filter with an extended allowlist scan, a generator split, per-shard piece export and
an exact cross-shard check (interval comparisons; negligible next to execution).

Budget: each shard is its own record input, cached by `run.py --resume` and run under the
4-process cap. Acceptance requires measured numbers, recorded in the PR, that meet the check
budget: every shard of H100 `k_stage_in` under 5 minutes and under the 16 GiB memory cap (the
GB10 8-row run is 293 s and about 14 GB, so the plan starts at 4 rows per shard, 12 shards, and
the row count is chosen from the first measurement). The total wall time at the cap is reported.
If a shard cannot meet 5 minutes at one row, the design is not accepted for that kernel and
option A (admitting CTAs as translates without executing them) is the next step.
