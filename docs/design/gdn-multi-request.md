# k_gdn with two or more requests: keep class mode

Status: design revision 2, for review before code. Owner: Coral SentryWard structural lane.
Builds on [relation-by-base.md](relation-by-base.md) (revision 4) and [distinct-rows.md](distinct-rows.md).
Budget: 900 s and 16 GiB per input.

## Finding

Class mode analyses only representative CTAs (17 for H100 `k_gdn`). It peaks at 4.2 GiB for any
request count. The per-unit re-run analyses every CTA and costs about 10 GiB per request. Every
H100 `k_gdn` shape leaves class mode today, at one of two sites:

| Shape | Restart reason | Site |
| --- | --- | --- |
| b1 | `footprint_symbolic_member:state` | footprint check, after the memory check passed |
| b2 and up | `cross_unit_unknown:symbolic_overlap:state` | cross-unit pass of the memory check |

The b1 reason is from the run record at 3f612ca (`unit_restart` in its report); the b4 reason is
from the 3f612ca diagnosis. Neither site changed in 829ee8e: the relation is read only by
`TryFamilyMemoryCheck`; the cross-unit lookup pass decides through `translatedFamilyOverlap`,
whose solver query has no row facts; and the footprint check is untouched. No b4 run at 829ee8e
was made: `pgrep -c -x sentryward-ptx` was 4. Two consequences:

- The family "pass 2" at 829ee8e (b1 110,752 families, b2 221,504) is the per-unit re-run's
  family check. Pass 1 is class mode's and does not grow with requests (59,812 families for b1
  and b2; no cross-request pairs).
- Fixing only the cross-unit site moves b2..b48 to the footprint restart, which b1 already hits.
  A must fix both sites.

## A. Class mode decides cross-request state pairs (decided: do it)

**A1, cross-unit.** In the address model's overlap decision for two translated family shapes,
before the solver: if each shape is `c + S t + r` with one lookup atom `t`, equal `S > 0`, residual
byte range `[lo, hi)` inside `[0, S)`, both atoms with premise bounds and `S * upper + hi < 2^63`
(checked 128-bit, the revision-4 window conditions), and `distinctRows` holds for the two atoms
(write row `i` against write or read row `j`, `i != j`), the result is Disjoint. The atoms are
the translated terms: in unit `u` the member reads element `arg + delta_u(table)`, the existing
class-mode fact the lookup pass already uses (`shift_terms`). One predicate, owned by the
address model, serves the family checker and class mode. Preconditions kept from the approved
designs: `upper` is the bound of each translated atom (never the representative's); the relation's
objects are read-only inputs under the tensor_properties contract; the window is exact (one
term, same width, no truncation); neither shape has an alias scope.

The cross-unit memo is scoped to one allocation and one ordered class pair. Within that scope the
buckets are fixed: lookup symbols, original arguments, coefficients, residual spans and
read/write roles. The relation configuration, tensor properties and address invariants are
launch-global. A group pair's decisions therefore depend only on the shifts it applies, and the
key must determine every input to every decision, solver results included:

- the relative allocation shift, as today;
- per common lookup object, the meet flag and meeting shift, as today;
- the absolute shifts, on both sides, of every object a bucket of either side looks up whose
  translated atom bounds or premises can vary by element: an object in a `distinct_rows` field,
  with a field policy, or pinned by an address invariant (term bound or joint injectivity);
- for a nonuniform class pair, the group indexes, as today.

Without the third item, two group pairs with one key could differ in row separation or atom
bounds, and a pair decided first would hide the other. Keys then grow from O(heads) to
O(units^2), at most 2.4 M at b48 (1,536 CTAs); each costs a few bucket pairs and no solver call
when rows separate, within the existing `unit_pairs` cap of 10 M.

Coverage. The lookup pass groups a class's units by their complete delta vector (the allocation,
then every looked-up object) and visits every ordered group pair, excluding only same-unit pairs
and the representative pair, which the in-class check decides. Every cross-unit unit pair lies in
exactly one group pair. A keeps that loop: the window rule only turns an Unknown decision into
Disjoint inside a visited pair, and the finer key only removes memo hits. No pair is skipped
that today is visited.

**A2, footprint.** The footprint check restarts on a symbolic offset whose atom has a field
bound, because a member reads another table element. Class mode may keep the representative's
verdict only when that verdict used no fact specific to the representative's element.

`Contain` decides a symbolic offset in two steps. First the interval hull: each atom over
`BoundsOf` (the symbol's object policy tightened by the field policies containing the element),
checked against the class's allowed range `[-min delta, extent - max delta)`. Second, if the hull
does not fit, `ExtremesOf` bisects with the solver over `Instantiate`, which adds address
invariants (term bounds on named elements), joint injectivity and injectivity. Only the first
step is policy-only. The verdict records its basis: `constant`, `policy_hull` or `solver`.

Rule: for `shift.units > 1` and any symbolic offset, keep class mode iff the basis is
`policy_hull` and, for every unit of the class and every atom of the offset, the translated
element `arg + delta_u(table)` lies in exactly the same set of field policies as `arg`. Equal
sets give equal `BoundsOf`, so each member's hull is the representative's hull moved by its
allocation delta, and the allowed range already covers every delta. Any other case restarts as
today. Cost: one policy-set comparison per (symbol, element, class), memoized.

The reviewer's launch shows why the basis matters: two identical one-thread CTAs each read 4 bytes
of `state[tab[blockIdx.x]]`, `state` is 4 bytes, both table elements have policy
`0 <= tab[i] <= 1`, and an address invariant gives `tab[0] = 0`. The hull is `[0, 8)`, so `Contain`
proves containment by the solver from `tab[0] = 0`; CTA 1 reads `[4, 8)`. Equal policy sets alone
would transfer the verdict. With the basis check it restarts and the per-unit run reports CTA 1.
The same launch with an object-wide policy instead of a field policy passes today: the current
check restarts only when an atom is field-bounded or has no hull, so it transfers a solver verdict
for an object-bounded atom. The rule above applies to every symbolic offset and closes both.

Alternative kept for later: re-run `Contain` once per distinct translated atom tuple (at most one
per request) when the basis is `solver`. Built only if the b1 class attempt shows `k_gdn`'s state
reads decided by the solver.

Obligations:

1. Row derivation exact, as in distinct-rows.md; translated arguments come only from the class's
   delta table (no new fact).
2. Window conditions per translated atom as revision 4; otherwise the solver path is unchanged.
3. The memo key determines every decision input, including solver queries; a key never stands
   for a pair whose result could differ.
4. Same row (in-place update) and read/read pairs are never separated; the rule only returns
   Disjoint, never a race or Unknown.
5. A2 transfers only a `policy_hull` verdict, with policy-set equality over all field policies of
   both elements.

Controls (fork tests, through the whole pipeline):

- A two-request `k_gdn`-shaped kernel decided in class mode (`unit_restart` absent, `race_free`);
  without the relation it restarts as today.
- A same-row cross-CTA overlap: the class-mode pass reports it or restarts, never `race_free`.
- The reviewer's launch above (pinned `tab[0] = 0`, `tab = [0, 1]`), once with a field policy and
  once with an object-wide policy: each must restart and then report CTA 1 past extent. Mutants:
  accept a `solver` basis; keep today's field-bounded-only restart. Each must give a false pass.
- Memo audit (test build): every memo hit is decided again and must equal the first decision for
  its key, status and solver verdict included. One control per key component (relative shift,
  meet shift, absolute relation shift, absolute field-policy shift, absolute pinned shift), each
  built so two group pairs agree on all other components and differ in result. Mutant: drop that
  component; the audit and the control's verdict must fail.
- Mutants: separate `i == j`; accept differing field-policy sets (a control with per-row bounds
  `next[0] < 2`, `next[1] < 2^40` must restart); use the representative's atom bound in the
  window check.

## B. Family pair grouping (decided: specify, defer)

With A, pass 2 does not run for `k_gdn`, and class mode's pass 1 has no cross-request pairs. Pass 2
runs only on a per-unit path, which costs about 10 GiB per request, so it fits 16 GiB only at b1.
Grouping cannot make b2 and up fit there. Specified so it is ready if a shape must run per unit:

- Group families with a window form by (allocation, `S`, atom key); the atom key gives the row.
  Families without a form go in one ungrouped bucket.
- For each ordered group pair with a write side: if `distinctRows` holds for the two atoms, count
  all their family pairs as disjoint without visiting them. Otherwise enumerate family pairs as
  today (revision 4: `R` plus the HB query). Ungrouped families are enumerated as today.
- Coverage: each family pair is in exactly one group pair. Assert at debug build that visited plus
  skipped pairs equal today's count. Same-row and read/read groups are never skipped; the rule
  never emits a race.

Cost (829ee8e: 0.31 us per visited pair; b1 pass 2 has 369 M same-row pairs, b2 adds 1.01 G
cross-row pairs). Visited pairs today about `369 M N + 503 M N(N-1)`; grouped `369 M N` plus
`(2N)^2` group pairs:

| Shape | Pairs today | Time today | Pairs grouped | Time grouped |
| --- | --- | --- | --- | --- |
| b2 | 1.74 G | 534 s | 0.74 G | about 230 s |
| b4 | 7.5 G | about 2,300 s | 1.5 G | about 460 s |
| b48 | 1.15 T | about 99 h | 17.7 G | about 5,500 s |

Controls if built: the distinct-rows controls through pass 2, plus the coverage assertion with a
mutant that skips a same-row group (false `race_free` on the same-row control).

## C. Event-edge store (decided: defer)

The fix stores consecutive non-affine edges in one plain segment, about 220 B to 20 B per edge.
It acts only on the per-unit path. With A, H100 `k_gdn` b1..b48 never take it. At b1 per unit it
would cut event edges from 2.5 to about 0.25 GiB, but the HB edges (5.9 GiB) and value IR with
scalar fold (8.8 GiB) set the 10.4 GiB peak, so per unit stays about 8 GiB per request and b2
still does not fit. Revisit when a row that must run per unit (class mode off, or another restart
reason) sits within about 2 GiB of the cap. eager-t4096-r48 (timeout at 15.1 GiB) is the first
candidate to measure.

## Expected results with A

Estimates, not measurements. Memory: class-mode peak, 4.2 GiB measured for b1 and b4; the delta
tables and memo grow with CTAs (kilobytes to tens of megabytes). Time: b1 class attempt ended at
about 154 s at 3f612ca, where it previously went on to a 575 s per-unit run; b4's ended at 411 s in
the diagnosis run (8 sync threads, host load about 100).

| Shape | Time | Peak RSS |
| --- | --- | --- |
| b1 | about 150 to 250 s | about 4.2 GiB |
| b2 | about 200 to 300 s | about 4.2 GiB |
| b4 | about 250 to 450 s | about 4.3 GiB |
| b48 | open: 300 s if the pre-memory phase is flat in CTAs; about 4,000 s if linear | about 4.5 GiB |

## Acceptance and measurement plan

Acceptance is 900 s and 16 GiB per shape, measured after A1 and A2 land, never estimated.

1. Run only with no other `sentryward-ptx` on the host (`pgrep -c -x sentryward-ptx` is 0), with
   `run.py --rows sm121/k_gdn --timeout 900 --cap 16` and the exact fork commit, PTX and input
   hashes recorded.
2. Shapes b1, b2, b4, b8, b48. Per shape: wall time, peak RSS, phase profile (pre-memory, memory
   check, footprint, later phases), `unit_restart` absent or its reason, `unit_pairs`, memo size,
   and the count of footprint verdicts by basis.
3. A later restart (past the footprint check) is a finding: record its reason and site and stop.

Fallback if b48 exceeds the budget, by the phase that dominates:

- Pre-memory phase linear in CTAs: profile it and write a separate design for that phase; A
  stays, since it is needed for any request count.
- Cross-unit memo: compact the key by replacing absolute shifts with what decisions read (the row
  index inside each relation field or "outside", and the translated element's policy set), with
  the same memo audit and controls.
- Otherwise: report b48 as failing the budget with the largest passing shape, not as accepted.

## Open questions

1. Does class mode's pre-memory phase (b1 about 90 s, b4 about 330 s in the diagnosis runs,
   both at 8 sync threads) scale with CTAs or with host load? It decides b8 and up.
2. Are `k_gdn`'s state reads decided by the policy hull? The generated inputs give the tables
   field range premises (the slot count bounds each row), so the hull should fit; whether an
   address invariant also names the tables was not checked. The b1 basis count settles it. If
   the basis is `solver`, A2 restarts and the per-tuple alternative is needed.
3. Any later class-mode restart (value lifting) past the footprint check is unobserved for b2
   and up.
4. The memo growth at b48 (2.4 M keys) is an estimate; recorded in the b48 run.

## Model

Not applicable: no state, protocol or concurrency change. Each rule is a pure decision over
facts the checker already holds (delta table, field policies, relation, containment basis).

## Alternatives considered

| Design | Guarantees | Cost | Openness | Migration |
| --- | --- | --- | --- | --- |
| A1 + A2 in class mode (chosen) | full | constant memory in N | any relation kind added to `distinctRows` | about 2 days |
| B grouping on the per-unit path | full | 10 GiB per request | same | about 1 day; b2 only |
| C plus a cheaper per-unit HB store | full | about 6 to 8 GiB per request | none needed | weeks; b2 at best |
| Return unsupported on a class-mode restart (no re-run) | none for bN | none | none | hours; decides nothing |

A wins: only class mode keeps memory flat in request count, and the fact it needs (distinct
rows) is already an input premise. B and C shrink a path that is linear in requests.

## Openness, visibility, parameters

| Set or item | Choice | Reason |
| --- | --- | --- |
| relation kinds read by the window rule | closed (`distinct_rows`) | a rule is proved per kind |
| field-policy equality | over all policies of the symbol | any new policy kind is covered |
| containment basis | closed (`constant`, `policy_hull`, `solver`) | only `policy_hull` transfers |
| memo key fields | closed, in `cross_unit_races` | soundness depends on each field |

No spec functions (SentryWard is C++). Parameters: requests, positions and heads come from the
launch input per shape; no GPU model, world size or deployment enters any rule.

Files: the window rule in `symbolic_address_model` (beside `Distinct`); the containment basis in
`ContainmentVerdict`, set by `Contain`; the memo key and audit in `cross_unit_races`; field-policy
equality in `footprint_checker` with an address-model query.

Design doc: this note.
