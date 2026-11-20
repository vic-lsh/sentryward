# k_gdn with two or more requests: keep class mode

Status: design revision 1, for review before code. Owner: Coral SentryWard structural lane.
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
address model, serves the family checker and class mode.

The cross-unit memo key must change. With uniform lookups it holds the relative translation and
which lookups coincide, not absolute rows. Two unit-group pairs with one key could then differ
in row separation, and a separated pair decided first would hide a same-row pair. When the
allocation has a lookup on a relation object, the key also holds both groups' absolute shifts
on those objects. Keys then grow from O(heads) to O(units^2), at most 2.4 M at b48 (1,536 CTAs);
each costs a few bucket pairs and no solver call when rows separate, within the existing
`unit_pairs` cap of 10 M.

**A2, footprint.** The footprint check restarts on a symbolic offset whose atom has a field bound,
because a member reads another table element, whose bound may differ. Keep the representative's
range when the bounds are equal: for every unit of the class, the element `arg + delta_u(table)`
lies in exactly the same set of field policies as `arg`. Equal sets give equal atom bounds,
hence the representative's hull shifted by the unit's allocation delta, as for a constant offset.
Otherwise restart as today. Cost: one check per (lookup symbol, element, class), memoized.

Obligations:

1. Row derivation exact, as in distinct-rows.md; translated arguments come only from the class's
   delta table (no new fact).
2. Window conditions per shape as revision 4; otherwise the solver path is unchanged.
3. Memo key includes absolute shifts on relation objects; a key never stands for a pair whose
   separation differs.
4. Same row (in-place update) and read/read pairs are never separated; the rule only returns
   Disjoint, never a race or Unknown.
5. Field-policy equality is over the full policy set of both elements; any difference restarts.

Controls (fork tests, through the whole pipeline): a two-request `k_gdn`-shaped kernel decided in
class mode (`unit_restart` absent, `race_free`); without the relation it restarts as today. A
same-row cross-CTA overlap: the class-mode pass reports it or restarts, never `race_free`. Mutants:
drop absolute shifts from the memo key (a control ordering a separated pair before a same-row pair
must then give a false `race_free`); separate `i == j`; accept differing field-policy sets (a
control with per-row bounds `next[0] < 2`, `next[1] < 2^40` must restart).

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

Memory: class-mode peak, 4.2 GiB measured for b1 and b4; the delta tables grow linearly in CTAs
(kilobytes). Time: b1 class attempt ended at about 154 s at 3f612ca, where it previously went on to
a 575 s per-unit run; b4's ended at 411 s in the diagnosis run (8 sync threads, host load about 100).

| Shape | Time | Peak RSS |
| --- | --- | --- |
| b1 | about 150 to 250 s | about 4.2 GiB |
| b2 | about 200 to 300 s | about 4.2 GiB |
| b4 | about 250 to 450 s | about 4.3 GiB |
| b48 | open: 300 s if the pre-memory phase is flat in CTAs; about 4,000 s if linear | about 4.5 GiB |

## Open questions

1. Does class mode's pre-memory phase (b1 about 90 s, b4 about 330 s in the diagnosis runs,
   both at 8 sync threads) scale with CTAs or with host load? It decides b8 and up. Measure b8 and
   b48 class attempts with profiles after A1 and A2 land.
2. Any later class-mode restart (value lifting) past the footprint check is unobserved for b2
   and up.
3. The memo-key growth at b48 (2.4 M keys) is an estimate; record `unit_pairs` in the b48 run.

## Model

Not applicable: no state, protocol or concurrency change. Each rule is a pure decision over
facts the checker already holds (delta table, field policies, relation).

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
| memo key fields | closed, in `cross_unit_races` | soundness depends on each field |

No spec functions (SentryWard is C++). Parameters: requests, positions and heads come from the
launch input per shape; no GPU model, world size or deployment enters any rule.

Files: the window rule in `symbolic_address_model` (beside `Distinct`); the memo key in
`cross_unit_races`; field-policy equality in `footprint_checker` with an address-model query.

Design doc: this note.
