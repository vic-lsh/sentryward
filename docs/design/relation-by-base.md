# Deciding a family relation by its symbolic bases (k_gdn)

Status: design, for review before code. Owner: Coral SentryWard structural-mode lane.

## Problem

`TryFamilyMemoryCheck` declines a whole launch when one candidate family relation's sampled
witness pair is disjoint (`family_memory_checker.cpp:442`, "Family Unknowns require further
partitioning; retain the exact existing path"). The fallback, `RunQuotientMemoryCheck`, tests
every access pair. For Coral's `k_gdn` (H100, one CTA of 128 threads) that is about 16,000 reads
of `state` at base `src_tab[s] * entry_units * 256` against 16,000 writes at base
`dst_tab[s * positions] * entry_units * 256`, and a single CTA does not finish in 900 s
(profile: 86% in the pairwise loop). The two bases are different load atoms, so the families'
symbolic offset keys differ and no existing rule relates them.

## Decision rule (revision 2, after review 1)

Scope: two access families on the **same resolved allocation** (equal identity, not merely
equal names; a shared alias scope or an unresolved identity keeps today's unknown/fallback,
`symbolic_address_model.cpp` ~698).

Each family's address must be, as the PTX computes it (bit-vector, every intermediate width and
signed conversion included), `c + S * t + a(m)`: `t` one integer atom, **the same atom for every
member of the family** (same canonical key; a family whose members load different words does not
qualify), `S` a constant, `a(m)` affine in the member index, `c` a constant. The checker derives
this form from the canonical address DAG only through exact steps and declines otherwise:

- Truncations and sign extensions are exact only under a bound. Coral's `recurrent.cu:61`
  computes `src_u = (int32_t)((int64_t)tab[s] * entry_units)` and then `src_u * 256 + rec_off`.
  With premises `0 <= tab[s] <= state_units` (from `formats/state.toml`, already
  `tensor_properties`) and the concrete `entry_units` argument, the checker requires
  `state_units * entry_units < 2^31` (so the `int32_t` truncation is the identity and the value is
  nonnegative) and `256 * entry_units * state_units + span < 2^63`. Then `t = tab[s]`,
  `S = 256 * entry_units`. If any bound fails or is missing, the form is not derived (fallback).
  A table value such as `2^32` cannot occur under the premise; without the premise it stays
  undecided.

Per family, the byte span `[min_m (a(m) + c), max_m (a(m) + c) + width_m)` is computed over every
member with each access's full byte width, and must lie in `[0, S)`.

Relation of F1, F2 meeting the form with equal `S`:

- `t1 != t2`: the windows `[S t, S (t + 1))` differ and each family's bytes stay in its window,
  so no byte is shared.
- `t1 == t2`: the families have one base; the candidate pairs are exactly the member pairs
  `(m, n)` whose byte ranges `[a1(m) + c1, + w1)` and `[a2(n) + c2, + w2)` intersect, computed by
  the existing same-base family relation (which already covers all overlapping pairs and widths).
  Every candidate pair then goes through the existing race rule unchanged: a cross-actor pair is a
  potential race (equal offsets do not mean the same actor), and a same-actor pair still needs the
  happens-before order the checker requires today (`family_memory_checker.cpp` ~369).

So the rule replaces only the decision "these two families' symbolic keys differ, so the
relation is unknown" by "the relation is the `t1 == t2` same-base relation"; it adds no new race
exemption. A shape that does not meet every condition keeps the fallback.

## Soundness obligations (for the reviewer)

1. Form derivation: the derived `c + S t + a(m)` equals the PTX bit-vector value for every
   member under the stated premise bounds (truncation, sign extension, wrap); otherwise no form.
2. Atom invariance: one canonical atom key per family, checked over every member.
3. Window separation for `t1 != t2` (exact integers under the bounds).
4. The `t1 == t2` case is the existing same-base relation and race rule, including ordering.
5. Same resolved allocation only; alias scopes and unknown identities untouched.

## Controls

- `k_gdn`-shaped kernel, read and write same affine map, two table bases: race free, decided
  without fallback.
- Same, but the write uses `a(m) + 4` (a neighbouring element of another thread): race reported
  (the `t1 == t2` relation now has cross-actor pairs).
- Two threads writing the same offset of the same base: race reported (equal offsets, two actors).
- Members of one family loading different table words (atom varies): not decided (fallback).
- Unequal access widths overlapping across an element boundary: decided by byte ranges, race
  reported where bytes intersect.
- A table premise allowing `tab * entry_units >= 2^31` (truncation not exact): fallback.
- Two allocations sharing an alias scope: fallback.
- A span larger than the stride (`bytes` past `S`): falls back (not decided by this rule).
- A base with no premise bound (wrap possible): falls back.

## Cost

One more exact linearization step in the canonical writer and one relation case in the family
checker; no change to the pairwise path.
