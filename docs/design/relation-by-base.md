# Deciding a family relation by its symbolic bases (k_gdn)

Status: design revision 3 (after review 2), for review before code. Owner: Coral SentryWard structural-mode lane.

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

Relation of F1, F2 meeting the form with equal `S`. The rule does not assert `t1 == t2` or
`t1 != t2`; it decides the pair under both cases and accepts only if both are decided:

- Case `t1 != t2`: the windows `[S t, S (t + 1))` differ and each family's bytes stay in its
  window, so no byte is shared. No query is needed.
- Case `t1 == t2`: F2 is **rebased** onto F1's atom: its offsets are rewritten from
  `c2 + S t2 + a2(m)` to `c2 + S t1 + a2(m)`, giving two families with one symbolic base key. The
  existing same-base family query (`family_memory_checker.cpp`, the path that today handles
  families with equal keys) then runs on the rebased pair, unchanged: it computes the member pairs
  `(m, n)` whose byte ranges `[a1(m) + c1, + w1)` and `[a2(n) + c2, + w2)` intersect and sends
  each through the existing race rule (a cross-actor pair is a race; a same-actor pair needs the
  happens-before order required today, ~369). The rebased families are used only for this query;
  the original families, their keys and the sampled witness are not modified, and no fact
  `t1 == t2` is added to any other query.

Outcome: the pair is race free only if the `t1 == t2` query is race free; a race in that query is
reported as a race (with the rebased witness, stating the case `t1 == t2`), and an unresolved
query keeps today's fallback. This replaces the current decision at :442 (decline because the
sampled witness is disjoint and the keys differ) for qualifying pairs only; it adds no new race
exemption.

## Soundness obligations (for the reviewer)

1. Form derivation: the derived `c + S t + a(m)` equals the PTX bit-vector value for every
   member under its own atom's premise bound (truncation, sign extension, wrap), checked in
   overflow-safe arithmetic; otherwise no form.
2. Atom invariance: one canonical atom key per family, checked over every member.
3. Window separation for `t1 != t2` (exact integers under the bounds).
4. Case split: `t1 != t2` is disjoint by windows; `t1 == t2` is decided by the existing same-base
   query on the rebased pair, including ordering; the equality is local to that query.
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
- Source bound tight but destination bound loose (review 2: `state_units = 1`, `entry_units =
  2^30`, `next < 3`): the destination family has no form, fallback.
- `t1 == t2` query finds a cross-actor overlap while the sampled witness is disjoint: race
  reported, not dismissed.
- Two allocations sharing an alias scope: fallback.
- A span larger than the stride (`bytes` past `S`): falls back (not decided by this rule).
- A base with no premise bound (wrap possible): falls back.

## Cost

One more exact linearization step in the canonical writer and one relation case in the family
checker; no change to the pairwise path.
