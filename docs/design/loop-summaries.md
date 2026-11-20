# Loop summaries for footprint containment

Status: design, for review before implementation (Coral D253 follow-up 3). Fork branch
`claude/sw-loop-summaries`, based on `claude/sw-path-conds` (bfcdd5a).

## Problem

The PTX frontend runs each lane as a concrete interpreter over partly symbolic values. A loop whose
exit test depends on loaded data stops it with `control_predicate_not_static`. Choice trees
(`branch_decisions`) cannot decide such a test: one choice forces every visit of the line, so
"taken" loops until `execution_step_budget_exceeded` and "not taken" runs exactly one iteration.

Coral rows blocked by this:

- `k_conv` (sm121 `conv.cu`): `for (i = 0; i < ends[s] - ends[s-1]; i++)` reads `qkv` row
  `ends[s-1] + i` and writes `outp` at the same row. nvcc -O3 emits it as a main loop unrolled by 4
  (counter stepping toward zero, exit test `setp.ne c, 0`) plus a remainder loop with its own counter.
- `k_sample`'s `row_of`: a binary search `while (hi - lo > 1) { mid = lo + (hi - lo) / 2; ... }`.

Footprint containment only. Race freedom is never claimed from a summarized loop.

## Goal and non-goals

- Decide containment of every access in every iteration of a counted loop whose trip count is
  loaded data, using the loop's own exit test as the bound on the iteration index.
- Not: race verdicts over summarized iterations, loops whose addresses depend on non-induction
  carried values (`row_of`), nested data loops, loops with shared-memory stores, synchronization or
  cross-lane operations.

## Mechanism

### 1. Eligible loops (static, per loop)

A natural loop `L` (existing `control_loops_`) whose data-dependent test classifies as a
computation loop (`ClassifyControl`: one latch, pre-test or post-test) is summarizable when:

- its only exit is that test (no `ret`/`exit`/`trap` and no other branch leaving `L`);
- it contains no nested loop, call, barrier (`bar.*`, `barrier.*`), shared-memory store, atomic or
  reduction, async copy, warp collective (`shfl`, `vote`, `match`, `redux`) or `membar`/`fence`;
- every register written in `L` (the carried set `C`) is a basic induction variable or is
  havocable (below).

Any other loop keeps today's behavior (the stop).

### 2. Induction variables

`r` in `C` is a basic induction variable (IV) when its only definition in `L` is
`add|sub.{s,u}{32,64} r, r, c` with `c` an integer known at loop entry, in a block that dominates
the latch. Its value at the start of iteration `k` is `r0 + s*k` with `s = ±c`, as a
`w`-bit value (wrapping; see 4). Every other register of `C` is havocked: a fresh unknown of its
declared width at the start of each summarized iteration.

### 3. Summary execution (per lane)

On reaching `L`'s header with an unknown test:

1. Make an iteration atom `k` (identity: loop header line, CTA, warp, lane, visit; bounds
   `[0, 2^31 - 1]`) and set each IV to `r0 + s*k` (an affine form over `k` and the entry value's
   atoms) and each havocked register to a fresh unknown.
2. Run one iteration along the stay path, under the path constraint of 4. Data branches inside the
   body behave as today (stop, or choice trees).
3. Leave by the exit edge with every register of `C` havocked (IVs included): values after the loop
   are not tracked.

Accesses recorded in step 2 carry the loop's constraint set. The template executor falls back to
per-lane execution at a summarized loop (as for `RefineGuard`).

### 4. The iteration constraint (relational)

Iteration `k` runs exactly when every test before it stayed. The summary records, for the body's
accesses, a linear constraint over atoms that holds for every executed `k`:

- `lt/le/gt/ge` test of an IV against a loop-invariant operand `n` (known, bounded atom, or an
  affine form over atoms), with the IV moving toward the bound: the stay condition at `k` (for a
  pre-test loop; at `k - 1` for a post-test loop, plus `k = 0` always runs once). Monotonicity makes
  "test stayed at `k`" imply it stayed at every `j < k`, provided the IV does not wrap before
  exiting; the summary adds the no-wrap condition `r0 + s*k` within the type's range as a
  constraint and refuses the loop if the solver cannot show `r0` and `n` lie in the comparison's
  domain (signedness as `RefineGuard`).
- `ne/eq` test of an IV against `n` (the -O3 counter `c != 0`): only when the solver shows
  `(n - r0)` is a nonnegative multiple of `s` (as an unbounded integer, under the input's premises);
  then the stay condition is `k < (n - r0) / s`. Otherwise refuse (the loop may wrap or not
  terminate).

The address model gains per-access constraint sets: a containment query asks Z3 for atom values
within their bounds, satisfying the access's constraints, with the address outside every declared
piece; unsat is contained, sat is a violation with a witness, unknown is undecided (never a pass).
The hull shortcut (`BoxOnly`) does not apply to an access with constraints.

### 5. Output and consumer contract

- `frontend_assumptions.loop_summaries`: per summarized loop, its header line, test line, IVs and
  havocked register count.
- `checks.footprint` is reported as today. The race analysis is not run on a launch with a
  summarized loop: `result` is `unsupported` with diagnostic `loop_summary_race_not_checked`, so a
  consumer that requires `race_free` does not accept it. Coral's `check_verdicts.py` already accepts
  `checks.footprint == passed` whatever the race result.

## Soundness argument (containment)

Fix a concrete execution, a lane and its `j`-th iteration of `L`. Map it to the summary with
`k = j`, each havocked register equal to its concrete value at that iteration's start, and the IVs
equal to `r0 + s*j` (true by induction: their only update adds `c` once per iteration on a block
every iteration passes). The iteration ran, so every earlier test stayed, which is the constraint
of 4 at `k = j`. Global loads in this frontend return values the analysis does not track (an
unknown, or a tensor-property atom over the loaded word), so the body's values are instances of the
summary's symbolic values; shared memory is not written in `L`, and no cross-lane operation is in
`L`, so no other lane's progress changes them. Hence every concrete access of that iteration is an
instance of a recorded summary access under its constraints, and an unsat outside query covers it.
After the loop every carried register is havocked, which over-approximates every exit state.

## Coverage of the Coral rows

- `k_conv` (gb10 PTX at the pinned source): the main loop's carried integers `%r50..%r53` each add
  4 per iteration and exit on `setp.ne %r50, 0`; the remainder loop's `%r57`, `%r58` add 1 and exit on
  `setp.ne %r57, 0`; both counters start from values derived from `n = ends[s] - ends[s-1]`. The
  carried floats (`t0..t2`, the window) are havocked; they are values, not addresses. With Coral's premises
  (`ends_tab` in `[0, 2048]`, `off_tab` in `[0, 1]`, from the monitor's J1/J4 checks) the qkv and
  outp rows are `ends[s-1] + i < ends[s] <= 2048`: contained if the solver proves the divisibility
  side conditions. This is the target and the first test.
- `k_sample` `row_of`: `lo` and `hi` are not IVs; havocking them leaves `mid` unbounded, so the read
  of `qo_indptr[mid]` stays not contained. Proving it needs a relational invariant
  (`lo < hi <= rows`), which this design does not infer. Recommended instead (Coral side, separate
  PR): replace the binary search with a scan of fixed trip count over the row capacity (a launch
  scalar), whose exit test is static; cost at most `r_cap` loads by one thread per emit beside a
  vocabulary scan of 248,320 lanes.

## Tests (test-first, native)

1. Counted loop `for (i = 0; i < n; i++) out[i] = in[i]`, `n` a loaded word bounded by a premise to
   the extent: contained.
2. Control: `i <= n` with the same premise: violation, witness at the last element.
3. The -O3 unrolled-by-4 plus remainder shape (compiled from CUDA in the test, as `k_conv`): contained.
4. Control: an address from a havocked carried register (`p = in[p]` chase): not contained.
5. Control: a loop with a shared-memory store or `bar.sync`: stop as today (not summarized).
6. Control: an `ne` test whose start is not a multiple of the step under the premises: refused.
7. A summarized launch reports `loop_summary_race_not_checked`, never `race_free`.

## Cost

Containment queries with constraints go to Z3 (no hull shortcut); each summarized loop adds one
atom and one constraint set per lane visit. The template executor runs per lane through the loop.
