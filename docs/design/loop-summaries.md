# Relational path conditions and loop summaries for footprint containment

Status: design, revision 4, for review before implementation (Coral D253 follow-up 3). Fork branch
`claude/sw-loop-summaries`, based on `claude/sw-path-conds` (bfcdd5a). Revision 1 drew R1..R10
(Codex review 1); revision 2 drew S1..S6 (Codex re-check). Each is cited where it is addressed.

Footprint containment only: every load and store of a launch stays inside its object's declared
bytes. Race freedom is never claimed from anything here.

## Problem

The PTX frontend runs each lane as a concrete interpreter over partly symbolic values. Two gaps
block Coral rows:

1. **Data branches lose their condition.** A choice tree (`branch_decisions`) decides a branch on
   loaded data, but the decided predicate only constrains later values in the narrow form of
   `RefineGuard` (one `setp` of an unknown register against a known integer, lt/le/gt/ge). An
   `eq`/`ne` test, a comparison of two unknowns, a mask (`n & 15`) or a combined predicate (`or.pred`)
   is dropped. Example: `k_sample` (sm121 `sample.cu`) branches on `or.pred (s >= slots, !last)`;
   forced not taken, `s` stays unbounded and the `next_tok[s]` store is reported outside (false alarm).
2. **Loops with a data trip count stop.** A choice on a loop test forces every visit. nvcc -O3 turns
   `for (i = 0; i < n; i++)` into a main loop unrolled by 16 with counter `-(n & -16)` and exit
   `setp.ne c, 0`, then tails of 8, 4 and a remainder loop on `-(n & 3)`, gated by data branches on
   `n & 15`, `n & 7`, `n & 3` (`tests/footprint_cases/src/loop.cu`). `k_conv` has this shape.

Stage A closes gap 1 and is a prerequisite of stage B (loops), whose trip counts and tails are
relations between masked values.

## Shared mechanism: a global fact set (R7)

Revision 4 replaces per-access constraint sets with one global fact set per run, which the address
model adds to every query, plus local sets only on accesses of summarized loop iterations (an extension of the existing
`address_invariants`, which are per-term only).

- A fact is a formula over atoms that is true of every concrete execution the run stands for (an
  execution following the leaf's branch choices). Atoms that denote one lane's values carry that
  lane's identity (CTA, warp, lane, visit), so facts from different lanes share only atoms that
  denote the same value in every lane (tensor-property elements, coordinates of the same thread).
  The conjunction of all recorded facts is then true of each such execution, under the assignment
  of every atom to its concrete value: one global environment is sound, and no global fact needs
  to be attached to an access. Atoms a loop exit havocks denote one concrete value each (the exit
  state), so facts on them are global like any other.
- Iteration facts are local, not global. A summarized loop's iteration atom `k` stands for every
  iteration, so `0 <= k < T` is not a fact about the execution: in the global set it would assert
  `T >= 1` and exclude zero-trip executions from every other query. Each access recorded inside a
  summarized iteration carries a handle to its local fact set (`0 <= k < T`), which its query adds to
  the global facts; no other access sees it. For any concrete iteration `j`, the assignment `k = j`
  (and the iteration's concrete values for its havocked registers) satisfies the local set together
  with the global facts at their concrete values. Global facts never mention iteration atoms (no
  internal branches in a summarized body, and exported values use `T`, not `k`). `T` and its facts
  are global: `T` is the trip count, which exists in every execution.
- Query: Z3 searches for atom values within their bounds, satisfying the facts that share atoms with
  the address (closure over shared atoms), with the address outside every declared piece of the
  access's direction. Unsat is contained; sat is a violation with a witness; unknown is undecided
  (never a pass). The fact set is checked satisfiable once per run; an unsat set (an infeasible
  leaf) reports `infeasible_path`, which a consumer treats as no evidence (never a pass) (S2).
- The global environment is fixed for the run, so the extrema cache stays keyed by offset for
  accesses without local facts; an access with local facts is keyed by (offset, local set) (R7).
  `Instantiate` adds atoms that occur only in facts. The `BoxOnly`, free-atom and constant-offset
  shortcuts may still pass an access whose unconstrained hull is inside (sound: facts only shrink
  the set); they may not report a violation for an access whose atoms occur in a fact. Accesses with
  different local sets never share a family. Representative CTA classes (`footprint_checker.cpp`)
  are disabled for a run with any fact, since a member's facts are not its representative's shifted.
  Each rule gets a native control.
- Atoms for loaded values: in a run with `branch_decisions` or a summarized loop, an integer load
  whose value would otherwise be unknown yields a fresh lane atom bounded by its type's range (exact
  identity: load line, CTA, warp, lane, visit), so values derived from it (`cvt`, `add`, `shl`)
  keep affine forms that later facts can constrain. Runs without choices and loops are unchanged.

## Stage A: relational path conditions

### A1. Symbolic integer terms

An integer register's value, when not a known integer, is an exact term: the atoms the frontend
already makes (tensor-property elements, path atoms, CTA/thread coordinates as integers), combined
by `add`, `sub`, `mul` by a constant, `shl` by a constant, `neg`, `cvt` between integer types, and
masks. Every term is evaluated with its machine width: the term denotes `v mod 2^w` (with the
signed reading where an instruction reads it signed), encoded exactly with Z3 integer `mod`, so no
wrap assumption is needed for constraints (S4 for stage A). A mask `x & -2^m` or `x & (2^m - 1)`
introduces a quotient atom `q` with `x' = 2^m q + r`, `0 <= r < 2^m` (`x'` the `w`-bit value of `x`),
and denotes `2^m q` or `r`. Any other operation (other bit ops, division, min/max, select on an
unknown predicate) yields an unknown, as today.

### A2. Recording decided branches

When a lane takes a branch whose predicate is unknown, by a user choice (`branch_decisions`), the
predicate's defining expression is recorded with the taken polarity: `setp.<cmp>.<int type>` of two
terms (all integer comparisons, `eq`/`ne` included, signedness from the type), and `and/or/xor/not`
`.pred` of such predicates, followed through registers within the lane's trace (the definition the
branch reads, found by the trace's origins, not by name: no scope ambiguity). A predicate with any
unknown leaf records nothing (sound: fewer constraints). Predicated non-branch instructions whose
predicate is unknown keep today's handling.

`RefineGuard` becomes a special case and is deleted when stage A lands (no old rule beside the
general one); its tests stay and must pass on stage A.

### A3. Soundness

A choice tree's leaf fixes one outcome per branch line for every lane (unchanged consumer condition
in Coral's `check_verdicts.py`). A concrete execution that follows the leaf's outcomes satisfies, at
each access, every recorded formula, because each formula is the exact machine meaning of the
predicate the lane branched on. So it is an instance of the access under its constraints.

## Stage B: loop summaries

### B1. Eligible loops (checked; a failed check refuses with `loop_summary_refused:<reason>`)

- `L` is a natural loop with one latch, classified as a computation loop by `ClassifyControl`
  (pre-test or post-test), whose only exit is its test; no `ret`, `exit`, `trap`, nested loop or call.
- The iteration CFG (`L` without the backedge) is acyclic and has no conditional branch other than
  the exit test (S6). A predicated instruction in `L` is allowed only if its predicate is defined
  outside `L` (loop-invariant), except memory instructions, which are recorded as executed
  (conservative: an access that may not happen is still checked).
- No barrier, atomic or reduction, async copy or TMA, warp or CTA collective, `membar`/`fence`,
  `mbarrier`, shared load or store, or `.local` store in `L` (R5, R6).
- Global stores in `L` only through pointers whose provenance is an object whose footprint is
  write-only and that no load of the launch reads (by provenance) (R6).
- Register discovery covers every destination, including vector, packed and predicate
  destinations; a register name declared in more than one lexical scope, or a write the decoder
  does not list as a destination, refuses (R4).

### B2. Induction variables (R4, S3)

`r` is an IV when it has exactly one definition in `L`, an unpredicated
`add|sub.{s,u}{32,64} r, r, imm` with a nonzero immediate literal `imm` (S3: no register step). Its
block dominates the latch, which in an acyclic iteration CFG with no internal branch means it runs
exactly once per iteration. Every other register written in `L` (predicates included) is havocked.

Update position: in a pre-test loop, every IV update must lie after the exit test (in the body or
latch); a loop with an update in the header prefix refuses (S1). In a post-test loop, the test may
read an IV before or after its update; the phase is fixed by instruction order.

### B3. Trip count, values and wrap (R1, R2, R3, S1, S2, S4)

The exit test is `setp.<cmp>.<int type> p, a, b` in the test block, read by the exit branch with its
polarity. One operand is an IV `x`; the other, `n`, is loop-invariant (a known integer or a term of
stage A over entry values). `eq`, float tests and predicate combinations refuse.

Let `U(k)` be the number of times an IV's update has executed when iteration `k` (0-based) is at a
given program point. Pre-test: at the header test of visit `k`, `U = k`; in the body and latch of
iteration `k`, `U = k` before the update and `k + 1` after it. Post-test: in the body of iteration
`k`, `U = k` before and `k + 1` after the update; the test of iteration `k` sees `x0 + imm*U` at its
phase. Values: `x0 + imm*U` as an exact integer.

Trip count `T` (body executions) is an integer atom with constraints, never a division:

- stay predicate `P(k)` is the test's comparison at the test's `U(k)`, read with the branch
  polarity and the stay edge, as an exact integer comparison (valid by the no-wrap facts below);
- pre-test: `T >= 0`, `P(j)` for `j < T` and `!P(T)`; with `P` monotone (proved: the IV moves toward
  `n` under the comparison's direction), this is `T >= 0 && (T == 0 || P(T - 1)) && !P(T)`;
- post-test: `T >= 1`, `(T == 1 || P(T - 2))` and `!P(T - 1)` likewise;
- `ne`: refused unless the solver proves `imm != 0`, `(n - x0)` an exact multiple of `imm` with
  quotient `>= 0` in the test's phase; then the equality form `imm * T' == n - x0` (with `T'` the
  `U` at the hitting test, `T = T'` pre-test or `T' + 1` post-test per phase) is used, and only then
  (S2). Monotone progress without wrap makes that test the first hit.

No wrap (S1, S4): the solver proves, under the facts recorded so far and `T`'s facts, that
for every executed update (pre-test: `U` in `[0, T]`, the final failing header visit included;
post-test: `U` in `[0, T]`), every IV value `x0 + imm*U` lies in its type's range in each reading the
code uses (signed or unsigned), and every `w`-bit address operation inside `L` built from IVs does
not wrap. Unknown or failed: refuse.

### B4. Summary execution (per lane)

1. Make the trip-count atom `T` (its facts global) and an iteration atom `k` (identity: header line,
   CTA, warp, lane, visit) with the local fact set `0 <= k < T`.
2. One iteration `k` (its accesses carry the local set): the header prefix (test block before the exit branch),
   then the body and latch, IVs at their `U` per program point, other carried registers havocked
   at iteration start. One atom covers the whole iteration, so values the prefix computes flow into
   the body consistently (R8).
3. Pre-test only, the final failing visit: the header prefix once more with every IV at `U = T` and
   other carried registers havocked; its accesses are checked; then the exit edge.
4. Exit: each IV at `x0 + imm*U_exit` (`U_exit = T`), exact by the no-wrap proof (which covers it);
   any IV the proof does not cover is havocked instead (S4). Every other carried register havocked.
   `T`'s facts stay in the fact set, so a remainder loop starting from the main loop's exit counter
   keeps the relation (R9).

The template executor falls back to per-lane execution before grouping any instruction of `L`.

### B5. Memory the loop reads (R6, S5)

The argument needs: a value the frontend models for a global load in `L` (an unknown, a
tensor-property atom, a pointer-table value) holds at every iteration. Global objects in this
frontend are either modeled by value (configured inputs, pointer arrays, tensor properties, which
`ValidateConstantInputs` validates as constant against writes it can resolve) or not modeled (a load
gives an unknown, which covers any value). Claim, by induction over the concrete execution's writes
in order: if the footprint verdict is `contained`, then no write lands outside its object, so no
write lands in a modeled read-only object; at the first write that would, all earlier modeled
values were correct, so the analysis represents that write soundly and reports it outside (or as an
unresolved or undeclared-object violation, which unresolved and generic writes already are), and
the verdict is not `contained`. Hence a `contained` verdict implies the modeled values held. Stage
B adds one check: a global load in `L` from an object some write of the launch (by provenance) also
writes refuses the loop. Coral's inputs declare every object a kernel writes (`footprints`), and the
pointer contract is `distinct_objects_noalias`; both are input facts, reported in the verdict.

### B6. Output and consumer contract

- `frontend_assumptions.loop_summaries` (header, test, IVs, `T` constraints, havocked count) and
  `loop_summary_refusals`; `path_constraints` counts per leaf.
- `checks.footprint` as today. With a summarized loop the race analysis is not run: `result`
  `unsupported`, diagnostic `loop_summary_race_not_checked`. Coral accepts `checks.footprint ==
  passed` whatever the race result.

### B7. Soundness (containment)

Fix a concrete execution following the leaf's choices, a lane and its iteration `j` of `L`. By B3,
`j < T` for body and latch accesses and `j <= T` for pre-test header accesses, and every IV equals
`x0 + imm*U(j)` at each program point (one unpredicated literal-step update per iteration, no wrap).
Other registers at iteration start take some value of their type (havoc). The iteration's path is
the unique path of the branch-free iteration CFG. Memory it reads holds its modeled values (B5), no
shared or local state is involved, and no cross-lane operation is in `L`. So each concrete access is
an instance of a recorded access under its constraints. After the loop, exported IVs are exact and
the rest havocked, over-approximating every exit state.

## Coral coverage (claims only after the tests pass)

- `k_conv`: target of test B3 below (pinned gb10 PTX, Coral's premises from #5217: `ends_tab` in
  `[0, 2048]`, `off_tab` in `[0, 1]`). No ordering premise on `ends`; the test shows whether the
  compiled form needs one. If its counter arithmetic is outside A1, the row stays refused and this
  note is revised with the gap.
- `k_sample`: stage A removes the `or.pred` false alarm at split 1; `row_of` (binary search, `lo`
  and `hi` not IVs) stays refused. Nothing here claims a replacement (R10).

## Tests (test-first, native; PTX from `tests/footprint_cases/src/loop.cu` at nvcc 12.9 -O3)

Stage A:
1. `RefineGuard`'s existing cases (guarded read, off-by-one, complement destination, shadowing)
   pass unchanged.
2. `eq`/`ne` and two-unknown comparisons constrain a later read (contained); off-by-one controls
   report outside.
3. `or.pred` of two comparisons forced not taken bounds both (k_sample's shape); control: `and.pred`
   forced not taken bounds neither.
4. Mask quotient: a read at `n & -16` with `n` bounded is contained; control with `n & -15` is unknown.
5. Infeasible leaf (contradictory choices) reports `infeasible_path`, never contained.
6. Address-model controls: a run with a fact does not use representative CTA classes; a shortcut
   never reports a violation for an access whose atoms occur in a fact; an atom that occurs only in
   a fact is instantiated.

Stage B:
1. `k_copy_n` (`i < n`, `n` bounded by a premise): contained (unrolled 16/8/4/1 shape).
2. Control `k_copy_le` (`i <= n`): violation at the last element.
3. `k_conv` pinned PTX with Coral's gb10 premises: contained (target).
4. Control `k_chase` (index from a havocked carried value): not contained.
5. Refusals, each with its reason: shared load; `.local` store; predicated IV update; register step;
   IV update in a pre-test header; vector destination redefining an IV; `eq` test; `ne` start not a
   multiple; a `u32` IV that wraps (`i < UINT_MAX` stepping 2 from `UINT_MAX - 1`); an exported IV
   that wraps; a load of an object the launch writes; an internal conditional branch.
6. `u32 i < n`, `n` up to `2^32 - 1`, an access past the extent only at high `k`: violation.
7. A pre-test header load runs on the failing visit: checked at `k = T`.
8. A summarized launch reports `loop_summary_race_not_checked`, never `race_free`.

## Cost and order

Constrained accesses go to Z3 with exact integer `mod` (slower than the hull); no family or CTA-class
aggregation for them. Stage A lands first (its own review of the implementation, its controls, the
Coral rows rerun), then stage B. Expect several days of fork work in total.
