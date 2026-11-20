# Loop summaries for footprint containment

Status: design, revision 2 (after Codex review 1: 6 P0, 3 P1, 1 P2, all addressed below; the
review's numbering is cited as R1..R10). For review before implementation (Coral D253 follow-up 3).
Fork branch `claude/sw-loop-summaries`, based on `claude/sw-path-conds` (bfcdd5a).

## Problem

The PTX frontend runs each lane as a concrete interpreter over partly symbolic values. A loop whose
exit test depends on loaded data stops it with `control_predicate_not_static`. Choice trees
(`branch_decisions`) cannot decide such a test: one choice forces every visit of the line, so
"taken" loops until `execution_step_budget_exceeded` and "not taken" runs exactly one iteration.

Coral row blocked by this: `k_conv` (sm121 `conv.cu`), `for (i = 0; i < ends[s] - ends[s-1]; i++)`
reading `qkv` row `ends[s-1] + i` and writing `outp` at the same row. nvcc -O3 emits a main loop
unrolled by 4 (counter stepping toward zero, exit `setp.ne c, 0`) and a remainder loop.

Footprint containment only. Race freedom is never claimed from a summarized loop.

## Scope (version 1)

A loop is summarized only when every condition below holds; any other loop keeps today's behavior
(the stop). Each condition is checked, never assumed; a failed check is a refusal with a named
diagnostic (`loop_summary_refused:<reason>` in the frontend facts), so a refusal can never pass.

## Mechanism

### 1. Eligible loops (static, per loop `L`)

- `L` is a natural loop (`control_loops_`) with one latch, classified by `ClassifyControl` as a
  computation loop, pre-test or post-test, whose only exit is that test (no `ret`, `exit`, `trap`,
  or other edge leaving `L`).
- No nested loop, call, barrier (`bar.*`, `barrier.*`), atomic or reduction (`atom`, `red`), async
  copy (`cp.async*`, `cp.async.bulk*`, TMA), warp or CTA collective (`shfl`, `vote`, `match`,
  `redux`, `bar.red`), `membar`/`fence`, or `mbarrier` operation in `L`.
- No store of any state space in `L` except global stores to objects whose footprint is
  write-only (direction `write`, read by no load of the launch) (R5, R6: a store the loop or another
  lane could read back would make a tracked value stale). No `.local` store and no shared load or
  store in `L` (R5, R6).
- Every global load in `L` reads an object declared read-only for the launch (footprint direction
  `read`, under the input's `distinct_objects_noalias` pointer contract) and checked so by the
  existing read-only validation (`public_input_memory.cpp`), run whether or not the race analysis
  runs (R6). Tensor-property atoms and pointer-table values over such objects are then stable for
  the whole launch.
- No data-dependent branch inside `L` other than the exit test (R8: a choice tree cannot cover
  every iteration's internal path). Branches whose predicates are known per lane are fine.
- Register discovery covers every destination of every instruction in `L`, including vector and
  packed destinations (`{a, b}`, `mov.b64 {lo, hi}`) and predicate destinations; any register name
  declared in more than one lexical scope, or written through a form the decoder does not list as a
  destination, refuses the loop (R4). The carried set `C` is every register written in `L`,
  predicates included.

### 2. Induction variables (R4)

`r` in `C` is an induction variable (IV) when it has exactly one definition in `L`, that definition
is an unpredicated `add|sub.{s,u}{32,64} r, r, c` with one scalar destination, `c` an integer known
at loop entry and nonzero, and its block lies on every path from the header to the latch (it
dominates the latch; with no nested loop and no internal data branch it then runs exactly once per
iteration). Its value at the start of iteration `k` is `r0 + s*k` as a mathematical integer, with
`s = ±c`, provided no wrap (3). Every other register of `C` (predicates included) is havocked.

### 3. Trip count and wrap (R1, R2, R3)

The exit test is `setp.<cmp>.<type> p, a, b` in the test block with `@p` or `@!p` on the branch.
Exactly one operand is an IV read at a fixed phase (before or after its update in that iteration;
the phase fixes the IV value the test sees as `r0 + s*k` or `r0 + s*(k + 1)`); the other is
loop-invariant (`n`: known, a premise atom, or an affine form over entry atoms). The stay predicate
`P(k)` is the comparison read with the branch polarity and the stay edge, interpreted in the
setp's signedness and width.

Supported comparisons: `lt, le, gt, ge` (and `lo, ls, hi, hs`) and `ne`. `eq` and any float or
predicate-combining form refuse. The summary computes, as mathematical integers under the input's
premises, a trip count `T >= 0` (pre-test: iterations whose test stays; post-test: body executions,
at least 1) with:

- relational test: `T` is the least `k >= 0` with `!P(k)` (pre-test) or `1 +` that over the
  post-update values (post-test), expressed in closed form (`ceil((n - r0) / s)` clamped at 0, by
  case on the comparison and the sign of `s`); the IV moves toward the bound or the loop refuses;
- `ne`: the solver proves `s != 0`, `(n - r0)` is a multiple of `s` with `(n - r0) / s >= 0`, and
  then `T = (n - r0) / s` (pre-test, test reading the pre-update value; the other phase/position
  combinations shift by one as above). Monotone, non-wrapping progress (next item) makes `T` the
  first hit, so no earlier modular hit exists.

No wrap: the solver proves that for every `k` in `[0, T]` (the values the test and body can see),
every IV used by an address, a comparison, or the exit test stays inside its type's range in the
interpretation each use reads (unsigned `[0, 2^w)`, signed `[-2^(w-1), 2^(w-1))`), and that the
`w`-bit operations computing addresses from IVs inside `L` do not wrap either. If any check is
unknown or fails, the loop refuses. With this, the mathematical and machine values agree on every
reachable iteration, and `T` covers all of them: no fixed bound on `k` (R2).

### 4. Summary execution (per lane)

On reaching `L`'s header with an unknown test:

1. Make an iteration atom `k` (identity: header line, CTA, warp, lane, visit), an unbounded integer
   with constraints. Set each IV to its affine form in `k`, havoc every other register of `C`.
2. Execute the header prefix (instructions of the test block before the exit branch, which
   `ClassifyControl` allows to read memory) under constraint `0 <= k <= T` for a pre-test loop (it
   also runs on the final, failing test) and `0 <= k < T` for a post-test loop; execute the body
   and latch under `0 <= k < T` (R8). Each access records the constraint of its program point.
3. Leave by the exit edge with each IV set to its exit value `r0 + s*T` (exact, by 3) and every
   other carried register havocked (R9: the remainder loop starts from the main loop's exit
   counter). `T` itself is a symbolic integer over entry atoms, carried as an affine form only when
   it is one (division results are kept as a fresh atom `q` with the defining constraint
   `s*q = n - r0`); otherwise the IVs are havocked too.

The template executor falls back to per-lane execution before grouping any instruction of `L`
(R8), as it does for `RefineGuard`.

### 5. Constraints in the address model (R7)

- Each recorded access carries a normalized constraint set (linear integer constraints over atoms).
  It survives lowering, compression and family extraction; accesses with different constraint sets
  are never merged into one family, and constrained accesses are not aggregated into representative
  CTA classes (each lane visit is checked on its own).
- Queries: Z3 asks for atom values within their bounds and satisfying the access's constraints with
  the address outside every declared piece of the access's direction. `Instantiate` includes atoms
  that appear only in constraints. Unsat is contained, sat a violation with a witness that satisfies
  the constraints, unknown undecided (never a pass).
- The extrema cache is keyed by (offset, normalized constraints). `BoxOnly`, the free-atom and
  constant-offset shortcuts may pass a constrained access only when the unconstrained hull is
  inside (safe); they never report a violation for a constrained access.

### 6. Output and consumer contract

- `frontend_assumptions.loop_summaries`: per summarized loop, header and test lines, IVs, `T`'s
  form, havocked register count; `loop_summary_refusals` with reasons.
- `checks.footprint` is reported as today. The race analysis is not run on a launch with a
  summarized loop: `result` is `unsupported`, diagnostic `loop_summary_race_not_checked`. Coral's
  `check_verdicts.py` accepts `checks.footprint == passed` whatever the race result; no consumer
  change is needed.

## Soundness argument (containment)

Fix a concrete execution, a lane, and its `j`-th iteration of `L`. By 3, `j < T` for body and latch
accesses and `j <= T` for pre-test header accesses, every IV equals `r0 + s*j` (one unpredicated
update on every iteration path, no wrap), so the machine values used by the iteration are the
summary's affine forms at `k = j`. Every other register's value at iteration start is some value of
its type, which the havoc covers. Memory the iteration reads is either a read-only global object
(stable for the launch by the validated read-only contract, so tensor-property atoms and pointer
tables hold) or not read at all (no shared, no local stores, no shared loads); no cross-lane
operation is in `L`. The iteration has no data-dependent branch other than the exit test, so its
path is the one the summary executed for that program point. Hence each concrete access is an
instance of a recorded access under its constraints, and the unsat outside query covers it. After
the loop, IVs hold `r0 + s*T` exactly and every other carried register is havocked, which
over-approximates every exit state; memory after the loop differs from entry only in write-only
objects, which no later load reads.

## Coral coverage

- `k_conv`: the target. Its claim is not made by this note: test 3 runs the pinned gb10 PTX of
  `k_conv` with Coral's premises (`ends_tab` in `[0, 2048]`, `off_tab` in `[0, 1]`, from the
  monitor's J1/J4 checks, Coral PR #5217) and must show contained; ordering `ends[s-1] <= ends[s]`
  is not a premise, so the proof must hold for any order (a negative `n` gives `T = 0` under the
  signed comparison, if that is the compiled form; the test shows which). If the compiled counter
  arithmetic (masks, shifts) is not affine in the supported way, the row stays refused and this note
  is revised with the gap (R9).
- `k_sample` `row_of` (binary search): not covered (`lo`, `hi` are not IVs). A replacement of the
  search in Coral's own kernel is a separate proposal with its own semantics, containment and
  latency evidence (R10); nothing here claims it.

## Tests (test-first, native)

1. Counted loop `for (i = 0; i < n; i++) out[i] = in[i]`, `n` a loaded word bounded to the extent:
   contained.
2. Control: `i <= n`: violation, witness at the last element.
3. The pinned `k_conv` PTX with Coral's gb10 premises: contained (target; see coverage).
4. Control: an address from a havocked carried register (`p = in[p]` chase): not contained.
5. Controls, each refused with its reason: a shared load in `L`; a `.local` store in `L`; a
   predicated IV update; a vector destination redefining an IV; an `eq` test; an `ne` test whose
   start is not a multiple of the step; a `u32` IV that can wrap (`i < UINT_MAX` stepping 2 from
   `UINT_MAX - 1`); a global store to an object some load reads; a data branch in the body.
6. A loop reaching `k = 2^31` (`u32` `i < n`, `n` up to `2^32 - 1`, an access past the extent only
   at high `k`): violation (no fixed bound on `k`).
7. A pre-test header load that runs on the failing test (`ld` before the exit branch reading
   `a[i]`): checked at `k = T`.
8. A summarized launch reports `loop_summary_race_not_checked`, never `race_free`.

## Cost

Constrained accesses go to Z3 (no hull shortcut for violations, no family or CTA-class aggregation);
each summarized loop adds one atom and up to two constraint sets per lane visit, and the template
executor runs per lane through `L`.
