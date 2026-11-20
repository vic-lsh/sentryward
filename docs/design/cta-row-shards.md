# Checking a large grid by CTA row shards (k_stage_in/out on H100)

Status: design revision 4 (after review 3), for review before code. Owner: Coral SentryWard structural-mode lane.

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
launch may be sharded only when its PTX passes an **allowlist** scan over the entry and every
function it reaches by direct call (an indirect call refuses). The scan is a new pass beside the
existing precondition scan (`unit_classes.cpp` PreconditionScan). An instruction is admitted only
if its opcode, every modifier and every operand kind are listed below; anything else refuses:

- Integer and predicate arithmetic: `add`, `sub`, `mul` (`.lo`, `.hi`, `.wide`), `mad`, `div`,
  `rem`, `min`, `max`, `neg`, `abs`, `and`, `or`, `xor`, `not`, `shl`, `shr`, `selp`, `setp`
  (integer comparisons), `cvt` between integer types, `mov` of registers, immediates and the
  special registers below. (Floating-point opcodes are not needed by the stage kernels; adding
  any opcode is a reviewed change to the list.)
- Special registers: only `%tid`, `%ntid`, `%ctaid`, `%nctaid`, `%laneid`. Every other special
  register (placement or time dependent: `%smid`, `%warpid`, `%clock*`, `%globaltimer*`, `%gridid`,
  `%cluster*`, `%pm*`, and the rest) refuses.
- Control flow: `bra` and `bra.uni` to a label of the same function (predicated or not), `ret`,
  `exit`, direct `call` to a function that passes the same scan.
- Memory: `ld.param`; `ld.global` and `st.global` with no ordering or scope qualifier (weak), any
  vector width, of a register address; `cvta.to.global`. Everything else in memory refuses,
  including generic-space `ld`/`st` and `cvta` to generic, `ld.global.nc`, `ld.volatile`,
  `.relaxed`/`.acquire`/`.release` forms, atomics and reductions of any space or scope, `prefetch`,
  `cp.async` of any form, TMA and `cp.async.bulk`, `st.async`, `mbarrier`, `multimem`, shared and
  distributed shared memory, `fence`, `bar`/`barrier` of any form, shuffles and votes.

The list is the minimum the admitted kernels need, not a model of what is safe in general; it is
widened only by a reviewed change with a control. Audit of the compiled H100 `k_stage_in`
(`stage.cu`, `nvcc 12.9`, `sm_90a`, 137 lines of PTX): opcodes `add.s64`, `sub.s64`,
`mul.lo.s64`, `mul.wide.u32`, `div.u32`, `div.u64`, `and.b64`, `or.b64`, `shl.b64`,
`cvt.u32.u64`, `cvt.u64.u32`, `cvta.to.global.u64`, `mov.u32` (of `%tid.x`, `%ntid.x`,
`%ctaid.x`, `%ctaid.y`, `%nctaid.x`), `setp.{eq,ne,lt,gt}`, `bra`, `bra.uni`, `ret`,
`ld.param.u64`, `ld.global.u64`, `ld.global.v4.u32`, `st.global.v4.u32`; no call. All are on the
list, so `k_stage_in` is admitted; the implementation re-runs this audit as a control on the
record's PTX. Global reads of a location some CTA of the launch writes are handled by the
cross-shard check below.

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

A frontend filter with the allowlist scan, a generator split, per-shard piece export and an exact
cross-shard check (interval comparisons; negligible next to execution).

Budget: the **complete uncached record** of one launch (every shard plus the cross-shard check)
must finish in under 10 minutes wall time using at most 4 concurrent `sentryward-ptx` processes
(the host-wide cap) and under the 16 GiB per-process memory cap. Estimate from the GB10 run (512 x
8 CTAs, 293 s, about 14 GB): H100 is 6 times the CTAs, about 1,760 process-seconds, so about 7.5
minutes on 4 processes at 4 rows per shard (12 shards, about 7 GB each). This is an estimate;
acceptance is the measured uncached wall time recorded in the PR. If it is over 10 minutes, the
design is not accepted for that kernel: the row stays trusted and option A (admitting CTAs as
translates without executing them) is the next step. Recording a longer total is not a pass.
