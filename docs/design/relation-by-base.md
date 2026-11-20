# Deciding a family relation by its symbolic bases (k_gdn)

Status: design revision 4 (after review 3), for review before code. Owner: Coral SentryWard structural-mode lane.

## Problem

`TryFamilyMemoryCheck` declines a whole launch when one candidate family relation's sampled
witness pair is disjoint (`family_memory_checker.cpp:442`, "Family Unknowns require further
partitioning; retain the exact existing path"). The fallback, `RunQuotientMemoryCheck`, tests
every access pair. For Coral's `k_gdn` (H100, one CTA of 128 threads) that is about 16,000 reads
of `state` at base `src_tab[s] * entry_units * 256` against 16,000 writes at base
`dst_tab[s * positions] * entry_units * 256`, and a single CTA does not finish in 900 s
(profile: 86% in the pairwise loop). The two bases are different load atoms, so the families'
symbolic offset keys differ and no existing rule relates them.

## Decision rule

Scope: two access families on the **same resolved allocation** (equal identity, not merely
equal names; a shared alias scope or an unresolved identity keeps today's unknown/fallback,
`symbolic_address_model.cpp` ~698).

Each family's address must be, as the PTX computes it (bit-vector, every intermediate width and
signed conversion included), `c + S * t + a(m)`: `t` one integer atom, **the same atom for every
member of the family** (same canonical key; a family whose members load different words does not
qualify), `S` a constant, `a(m)` affine in the member index, `c` a constant. The checker derives
this form from the canonical address DAG only through exact steps and declines otherwise:

- Truncations and sign extensions are exact only under a bound, derived **per table atom**, never
  shared between families. Coral's `recurrent.cu:61` computes
  `u = (int32_t)((int64_t)tab[i] * entry_units)` and then `u * 256 + off`. The bound on `tab[i]`
  is the premise of that table's own format row: the source table `carry_src` has
  `0 <= tab <= state_units`, the destination table `next` has `0 <= next < state_entries`
  (pad sinks included, `formats/state.toml` ~84). For each family, with `B` its atom's premise
  upper bound and the concrete `entry_units`, the checker requires `B * entry_units < 2^31` (the
  `int32_t` truncation is the identity and the value nonnegative) and
  `256 * entry_units * B + span < 2^63`, both evaluated in checked 128-bit arithmetic (an
  overflowing intermediate fails the bound). Then `t = tab[i]`, `S = 256 * entry_units`. If any
  bound fails, or the atom has no premise bound, that family has no form (fallback).

Per family, the byte span `[min_m (a(m) + c), max_m (a(m) + c) + width_m)` is computed over every
member with each access's full byte width, and must lie in `[0, S)`.

Relation of F1, F2 meeting the form with equal `S`. The rule **only proves a pair safe**; it never
reports a race or an unresolved verdict. Whenever it cannot prove safety, the pair takes today's
path unchanged (the decline at :442 and the exact fallback), so a race or an unresolved verdict is
only ever produced by existing code. It considers both cases of the atoms:

- Case `t1 != t2`: the windows `[S t, S (t + 1))` differ and each family's bytes stay in its
  window, so no byte is shared.
- Case `t1 == t2`: the member pairs that can share a byte are contained in
  `R = {(m, n) : [a1(m) + c1, + w1) and [a2(n) + c2, + w2) intersect}`, computed exactly from the
  within-window offsets (the same interval arithmetic as the same-base family relation).

So `R` over-approximates the pairs that can overlap in any execution, whether or not `t1 == t2`
is feasible under the premises. The pair (F1, F2) is safe if every pair in `R` is ordered by
happens-before.

Wiring into `TryFamilyMemoryCheck`. For a qualifying (F1, F2), the checker does not build the
key-based relation and does not call `memory.mayOverlap` on a sampled witness. It builds a
`FamilyRelationBatch` request for the **original** families (their actor, event and member
identities unchanged; only the requested member relation is `R` instead of the key-derived one)
and submits it to `hb.checkOrderBatch` with the other requests:

- `R` empty: no request; the pair is disjoint.
- The HB result leaves no unordered pair (`witness(domain)` is empty for every domain): safe.
- Any unordered pair remains (a cross-actor pair, or a same-actor pair without the order today's
  checker requires): the rule gives up on the whole launch, `declined = true` exactly as at :442,
  and the exact fallback decides. No race is reported from `R`.

No equality fact enters any query, the sampled witness and the address model are not consulted for
these pairs, and the HB query is the existing one on the original events. An infeasible
`t1 == t2` case can only make the rule give up (fallback), never produce a race.

## Soundness obligations (for the reviewer)

1. Form derivation: the derived `c + S t + a(m)` equals the PTX bit-vector value for every
   member under its own atom's premise bound (truncation, sign extension, wrap), checked in
   overflow-safe arithmetic; otherwise no form.
2. Atom invariance: one canonical atom key per family, checked over every member.
3. Window separation for `t1 != t2` (exact integers under the bounds).
4. `R` contains every member pair that can share a byte, in either case of `t1, t2`; the pair is
   accepted only when the existing HB query on the original families orders every pair of `R`.
   The rule emits no race and no unresolved verdict; any failure is today's decline.
5. Same resolved allocation only; alias scopes and unknown identities untouched.

## Controls

- `k_gdn`-shaped kernel, read and write same affine map, two table bases: race free, decided
  without fallback.
- Two threads writing the same offset of the same base: race reported (equal offsets, two actors).
- Members of one family loading different table words (atom varies): not decided (fallback).
- Unequal access widths overlapping across an element boundary: decided by byte ranges, race
  reported where bytes intersect.
- A table premise allowing `tab * entry_units >= 2^31` (truncation not exact): fallback.
- Source bound tight but destination bound loose (review 2: `state_units = 1`, `entry_units =
  2^30`, `next < 3`): the destination family has no form, fallback.
- Full path through `TryFamilyMemoryCheck` (not the classifier alone): the `k_gdn`-shaped kernel
  above is accepted with no fallback and no `mayOverlap` call on its state families.
- Cross-actor pair in `R` (write at `a(m) + 4`): the rule declines and the fallback reports the
  race; the rule itself emits nothing.
- Premises forcing `t1 = 0`, `t2 = 1` with a cross-actor pair in `R`: the rule declines (fallback);
  no race is reported by the rule.
- Two allocations sharing an alias scope: fallback.
- A span larger than the stride (`bytes` past `S`): falls back (not decided by this rule).
- A base with no premise bound (wrap possible): falls back.

## Cost

One more exact linearization step in the canonical writer, the `R` interval computation, and one
HB request per qualifying pair (member rows, not member pairs, as the existing batch); no change to
the pairwise fallback.
