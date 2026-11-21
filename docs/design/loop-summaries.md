# Relational path conditions and loop summaries for footprint containment

Status: design, revision 5, for review before implementation (Coral D253 follow-up 3). Fork branch
`claude/sw-loop-summaries`, based on `claude/sw-path-conds` (bfcdd5a). Revision 1 drew R1..R10
(Codex review 1); revision 2 drew S1..S6 (re-check 1); revision 3 drew T1..T6 (re-check 2). Each is cited where it is addressed.

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

An integer register's value, when not a known integer, is an exact term over atoms (tensor-property
elements, lane atoms of loaded values, path and iteration atoms, coordinates). Each operation is a
width-indexed transformation of unsigned residues; `u_w(v) = v mod 2^w` and the signed reading
`s_w(v) = u_w(v) - 2^w` when `u_w(v) >= 2^(w-1)`, else `u_w(v)` (T5):

| PTX | term |
| --- | --- |
| `add/sub.<t>w` | `u_w(a +/- b)` |
| `mul.lo.<t>w` by a known constant `c` | `u_w(a * c)` |
| `mul.wide.<u/s>w` by a known constant | `u_2w(r_w(a) * c)`, `r_w` = `u_w` for `.u`, `s_w` for `.s` |
| `shl.b<w>` by a known count `c` | `u_w(a * 2^c)` for `c < w`; `0` for `c >= w` (PTX clamps the count) |
| `neg.s<w>` | `u_w(-a)` |
| `cvt.<dst>.<src>` | widening: `u_dst(r_src(a))` (`r` = `s_src` when the source type is signed: sign extension, else `u_src`: zero extension); narrowing: `u_dst(a)` |
| `and.b<w>` with `-2^m` / `2^m - 1` (`m < w`) | quotient atom `q`, remainder atom `r`, facts `u_w(a) = 2^m q + r`, `0 <= r < 2^m`, `0 <= q < 2^(w-m)`; the term is `2^m q` / `r` |
| `mov`, `cvta` of an integer | unchanged |

`setp.<cmp>.<t>w a, b` compares `r_w(a)` and `r_w(b)` at its own width and signedness (`r_w` as
above), so a term wider than the comparison is read at the comparison's width. Any other operation
(other bit ops, division, min/max, `selp` on an unknown predicate, carry chains) yields an unknown, as
today. Each table row gets a boundary control (sign extension of `0xffffffff`, a shift by `w`, a
mask of a value already reduced mod `2^w`).

### A2. Recording decided branches

Predicate terms are built when the defining instruction executes, from its operand values, not
reconstructed from the trace afterwards (T4: trace origins forward through `cvt` and `mov` and lose
extensions). A `setp.<cmp>.<int type>` whose result is unknown and whose operands are terms (or
known) gives its destination predicate the term `cmp(r_w(a), r_w(b))`; a complement destination
(`p|q`) gets the negation; `and/or/xor/not.pred` combine the terms of their operands; any other
definition (a predicate loaded, moved through an integer, or combined with an unknown) has no term.
Predicate terms live in the lane state beside the register values and are overwritten with them.

When a lane takes a branch on an unknown predicate by a user choice (`branch_decisions`), the
branch's predicate term, with the taken polarity, is recorded as a global fact. No term: nothing is
recorded (sound: fewer facts). The template executor falls back to per-lane execution before every
chosen branch whose predicate is unknown (not only `GuardCompare`'s shape), so each lane's term is
its own. A later visit of the same line records its own fact; contradictory visits make the fact set
unsat, which reports `infeasible_path` (never a pass).

`RefineGuard` becomes a special case and is deleted when stage A lands (no old rule beside the
general one); its tests stay and must pass on stage A.

### A3. Soundness

A choice tree's leaf fixes one outcome per branch line for every lane and visit. Coral's consumer
condition (`check_verdicts.py` `choice_tree_reason`: every chosen line a branch in no CFG cycle, no
cross-lane operation, no call) still decides whether a set of leaves covers a launch; facts do not
relax it. A concrete execution that follows the leaf's outcomes satisfies, at
each access, every recorded formula, because each formula is the exact machine meaning of the
predicate the lane branched on. So it is an instance of the access under its constraints.

## Stage B: loop summaries

### B1. Eligible loops (checked; a failed check refuses with `loop_summary_refused:<reason>`)

- `L` is a natural loop with one latch, classified as a computation loop by `ClassifyControl`
  (pre-test or post-test), whose only exit is its test; no `ret`, `exit`, `trap`, nested loop or call.
- The iteration CFG (`L` without the backedge) is acyclic and has no conditional branch other than
  the exit test (S6). A predicated instruction in `L` is allowed only if its predicate is defined
  outside `L` (loop-invariant). If that predicate is known for the lane, the instruction runs as
  today. If it is unknown, its access (if any) is recorded as possibly executed and every destination
  is havocked, which covers both the executed value and the old value (T1).
- No barrier, atomic or reduction, async copy or TMA, warp or CTA collective, `membar`/`fence`,
  `mbarrier`, shared load or store, or `.local` store in `L` (R5, R6).
- Global stores in `L` only through pointers whose provenance is an object whose footprint is
  write-only and that no load of the launch reads (by provenance) (R6).
- Every global load in `L` is from an object covered by the immutability premise (B5).
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
- `ne`: refused unless the solver proves `imm != 0` and `(n - x0)` an exact multiple of `imm` with
  quotient `T' >= 0`, where `T'` is the number of updates when the test first sees `x0 + imm*T' = n`.
  Then `imm * T' = n - x0` (equality, used only here, S2) and, by case (T3):
  pre-test (update after the test, B2): `T = T'`; post-test with a post-update test: `T = T'` and
  `T' >= 1` required; post-test with a pre-update test: `T = T' + 1`. Monotone progress without wrap
  makes that test the first hit.

Existence (T6): before `T` is added, the solver proves, for every entry state satisfying the global
facts and atom bounds, that the closed-form count `T*` of the case (relational: `max(0, ceil((n -
x0) / imm))` adjusted for `le/ge` and post-test, as an integer function; `ne`: `(n - x0) / imm`)
satisfies `T`'s definition and the no-wrap condition below. A loop whose entry states admit no such
`T*` (a wrapping or nonterminating machine loop) refuses; `T`'s facts never make a feasible entry
state infeasible. Control: a loop whose mathematical trip-count constraints are unsat for a
reachable entry state refuses, rather than reporting `infeasible_path`.

No wrap (S1, S4): the solver proves, under the facts recorded so far and `T`'s facts, that
for every executed update (pre-test: `U` in `[0, T]`, the final failing header visit included;
post-test: `U` in `[0, T]`), every IV value `x0 + imm*U` lies in its type's range in each reading the
code uses (signed or unsigned), and every `w`-bit address operation inside `L` built from IVs does
not wrap. Unknown or failed: refuse.

### B4. Summary execution (per lane)

1. Make the trip-count atom `T` (its facts global) and an iteration atom `k` (identity: header line,
   CTA, warp, lane, visit) with the local fact set `0 <= k < T`.
2. One iteration `k` (its accesses carry the local set): the iteration CFG in its concrete order,
   from the header, IVs at their `U` per program point, other carried registers havocked at
   iteration start. Pre-test: the header prefix (test block before the exit branch), then the body
   and latch. Post-test: the body, then the test-block prefix (U2). One atom covers the whole
   iteration, so values flow between program points consistently (R8). Mask definitions over an
   IV or any value that depends on `k` add their quotient/remainder facts to the iteration's local
   set, not the global one (U4).
3. Pre-test only, the final failing visit: the header prefix once more with every IV at `U = T` and
   other carried registers havocked; its accesses are checked; then the exit edge.
4. Exit: each IV at `x0 + imm*U_exit` (`U_exit = T`), exact by the no-wrap proof (which covers it);
   any IV the proof does not cover is havocked instead (S4). Every other carried register havocked.
   `T`'s facts stay in the fact set, so a remainder loop starting from the main loop's exit counter
   keeps the relation (R9).

The template executor falls back to per-lane execution before grouping any instruction of `L`.

### B5. Memory the loop reads (R6, S5)

The argument needs: a value the frontend models for a global load in `L` (a tensor-property atom, a
pointer-table value, a configured input) holds at every iteration. Containment does not establish
this (T2: a `readwrite` object can be overwritten within its bytes, and a wrongly modeled earlier
read can misplace the first clobbering write). It is a premise instead: **modeled inputs are
immutable during the launch**. SentryWard's input already states it for configured inputs (the
fixed-read-only caller contract, `public_input_memory.cpp`), and Coral's generator emits premises
only for images no kernel writes (its formats are read-only metadata checked before the step).
Stage B makes the premise explicit and narrow:

- a summarized loop may load only from objects declared `read` (not `readwrite` or `write`) in the
  input's footprints, and a load in `L` from any other object refuses the loop;
- the verdict lists the premise (`frontend_assumptions.modeled_inputs_immutable`, with the objects a
  summarized loop read), so the consumer states it;
- unmodeled loads (an unknown or a fresh lane atom) need no premise: their values are unconstrained
  by the analysis except through recorded facts, which are about the value actually loaded.

Coral side: the premise is the same one its tensor-property premises already rest on (the monitor
checks each image before the step and no kernel writes it); `check_verdicts.py` must accept a
footprint verdict with `modeled_inputs_immutable` only for objects whose declaration is read-only.

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
the unique path of the branch-free iteration CFG. Memory it reads holds its modeled values (B5, by the immutability premise), no
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

## Re-check 3 (U1..U6) and stage A as implemented

- U1 (P0, stage A): a `setp` with a Boolean modifier (`setp.lt.and p|q, a, b, c`) gets no predicate
  term (only the plain form does); a control covers it in the implementation review.
- U2 (P0, stage B): B4 executes the iteration CFG in its concrete order for each shape (post-test:
  body before the test-block prefix); a control writes, in the post-test prefix, a register the body
  read. To be written into B4 before stage B starts.
- U3 (P0 conditional): the consumer establishes immutability of each reported object for the whole
  launch independently (Coral: the image is metadata the monitor checks before the step and no
  kernel writes; a read-only footprint declaration alone does not discharge it).
- U4 (P1, stage B): mask definitions of an IV inside a summarized iteration are local facts.
- U5 (P1): the consumer requires `reads == contained` (Coral's `check_verdicts.py` already does:
  both halves contained), so the `noninterference` read mode cannot pass an outside read.
- U6 (P2): a satisfiable query is a potential violation with a symbolic witness.

Stage A implementation choices (fork commit after 798dcbb):
- Facts on ONE atom with a lane identity are not invariants: they tighten that atom's interval
  (the lane keeps the tightest version per atom and replaces it in every register holding it), so
  the address model still decides such accesses by the hull. Facts over two or more atoms become
  term invariants (`address_invariants` `term` with bounds), written with the tightest atoms.
- The address model propagates bounds over term invariants (interval fixpoint), so the footprint
  hull and the declared-pieces check see relational facts; a single-row piece contains any hull
  inside it.
- An unsigned ordering of a value just below zero (`w - 1` with `w >= 0`) against a known `c` is
  exact as `0 <= v < c` (or `v > c || v < 0`) when every negative value wraps above `c`.
- Choice runs execute lane by lane with class mode off (lane atoms are per lane); k_feed's leaf
  takes about 95 s (was about 5 s).

## Stage C: disjunctive facts as case splits (design, for review)

Problem. A chosen branch's meaning is recorded only where it is a conjunction. An unsigned ordering
of a value whose range crosses zero is exactly a disjunction: on 32 bits, `u(v) <u c` is
`(v >= 0 && v < c) || (v < 0 && v + 2^32 < c)`. Compiled `k_conv` guards its unrolled loop with
such a compare (`setp.gt.u32 p, lo - hi + 1, 0xFFFFFFFC`); its not-taken side is the fact that
makes the loop's trip count provably nonnegative. Today it records nothing, so the loop is not
summarized.

Mechanism.
- C1. Exact disjunctive terms. `CompareTerm`'s wrapped fallback (an unsigned ordering whose operand
  `v` has range `[L, U]` with `-2^w <= L < 0 <= U < 2^w` and the other operand a known `c`) gives
  the exact term `Or(And(v >= 0, v REL c), And(v < 0, v REL c - 2^w))`, simplified when a disjunct is
  empty on `[L, U]` (today's rule is the case where the second disjunct is empty). Other forms are
  unchanged.
- C2. Cases. At a chosen branch, the asserted formula (term with the taken polarity) is put in
  disjunctive normal form over its literals, at most 8 disjuncts (more: no fact, as today). One
  disjunct: its literals are facts, as today. Several: the input's `fact_cases` (a list of
  `{line, case}`) names the disjunct for that branch line; its literals are recorded. With no
  entry for the line, the run stops with diagnostic `fact_disjunction` (details: line, `cases`, the
  number of disjuncts), like `control_predicate_not_static`.
- C3. Infeasible cases. A case whose facts contradict the run's other facts makes the fact set
  unsat: the run reports result `infeasible_path` (the address model's consistency check proved
  unsat; unknown is never reported as infeasible).

Coverage. A case split is a choice like a branch choice: one case per line for every lane and
visit, so leaves combine into a launch verdict under the same consumer condition (Coral
`check_verdicts.py` `choice_tree_reason`: chosen lines in no CFG cycle, no cross-lane operation,
no call). Each execution's lane satisfies one disjunct at each such branch, hence follows one leaf;
an `infeasible_path` leaf stands for no execution and covers nothing that needs covering.

Coral side. `run.py` expands `fact_disjunction` like a data branch, into one child per case
(`fact_cases`), counted against the same depth and run budgets. `check_verdicts.py` accepts a leaf
with result `infeasible_path` as covering no launch (a proved-unsat fact set), and counts
`fact_cases` lines with the branch choice lines in `choice_tree_reason`.

Tests (test-first). (1) The wrapped compare's two disjuncts each give their facts in their case;
(2) k_conv's guard shape: case split lets the loop summarize and the leaf is contained (target);
(3) control: a case whose facts contradict an earlier fact reports `infeasible_path`, not contained;
(4) control: without `fact_cases` the run stops with `fact_disjunction`, never contained;
(5) a disjunction with more than 8 disjuncts records nothing.

## Lane-scoped facts (Codex review of stage C: P0 for stages A and C)

Finding. One global fact set is unsound for choice trees. A leaf forces one outcome per line on
every lane, but in a real execution lanes take different outcomes. Lane A's access in execution E
is covered by the leaf with A's own choices; in that leaf, another lane B's facts (B forced onto
A's choices) may constrain an input element A reads, excluding E from A's query. Likewise a leaf
whose global fact set is unsat may still be the leaf that covers A.

Rule. Every path fact carries the identity of the lane that recorded it (CTA coordinates, warp,
lane). A containment query for an access uses the input's invariants, the facts of the access's
own lane, and (stage B) the local facts of iteration atoms the access names. Per-lane coverage:
lane A's facts are the conditions A's own path satisfies in E, so every concrete access of A in E is
an instance of the leaf's access under A's facts. If A's facts are proved unsat, no execution has A
on this path: A's accesses in this leaf are vacuous (reported, counted, not violations). Unknown
satisfiability is undecided, never vacuous. There is no leaf-level infeasible_path.

Consequences. The global consistency check and global bound propagation cover only the input's
invariants; per-lane consistency and propagation are computed for the lane of the query (cached
per lane). Stage C's cases are chosen per line like branch choices and their facts are lane facts;
a case unsat for one lane is vacuous for that lane only.
