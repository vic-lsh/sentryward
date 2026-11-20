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

## Decision rule

Write each family's footprint offset as `c + S * t + a(m)`: `t` an integer atom (a loaded word),
`S` a constant coefficient, `a(m)` affine in the member index `m` (thread and unrolled index),
`c` a constant. Two families F1, F2 on the same allocation with the same `S`, and with each
family's per-entry span inside one stride (`min_m a(m) + c >= 0` and
`max_m (a(m) + c) + bytes <= S`, computed exactly over the members), relate as follows:

- `t1 != t2`: the accesses lie in different stride-aligned windows `[S t, S (t + 1))`, so they
  are disjoint. No premise is needed beyond integrality of `t` (a loaded integer word).
- `t1 == t2`: the relation is the existing affine member relation of two families with one
  base: pairs `(m, n)` with `a1(m) + c1` overlapping `a2(n) + c2`. The current code already
  decides this exactly for families with equal symbolic keys; reuse it.

So the relation is the union of the `t1 == t2` affine relation and nothing else. When that
relation contains only same-actor pairs (as in `k_gdn`, where read and write offsets are the
same affine map of the same thread and index), the family pair is race free, whatever the
tables hold. Any family pair that does not meet the shape (different `S`, a span exceeding
`S`, a non-affine member map, a non-integer or unbounded atom) keeps today's behavior.

## Where `S * t` comes from

Today the canonical address is one opaque atom per access. The strided-families branch
(`claude/sw-strided-families`, 6e012a7) already peels coordinate-only constants; this design
needs one more exact step in the same writer: a product of an opaque subterm and a constant
(the launch argument `entry_units` is a constant in the input, 256 is a literal) becomes
`S * atom`, exact modulo `2^64`. Wrap: `S * t + rest` must not wrap, which holds when `t` is
bounded by a premise (`tab` elements carry `tensor_properties` bounds from `formats/state.toml`)
and `S * t_max + span < 2^63`; otherwise the term stays opaque (fallback).

## Soundness obligations (for the reviewer)

1. Integrality and window separation: `t1 != t2` and both spans within `[0, S)` imply the byte
   ranges are disjoint. Exact integer arithmetic, no wrap under the stated bound.
2. The `t1 == t2` case reduces to the existing same-base relation; its correctness is the
   existing code's.
3. The rewrite of `load * const` into `S * atom` is exact under the wrap bound; otherwise refused.
4. Nothing undecided is accepted: every non-matching shape keeps the pairwise fallback.

## Controls

- `k_gdn`-shaped kernel, read and write same affine map, two table bases: race free, decided
  without fallback.
- Same, but the write uses `a(m) + 4` (a neighbouring element of another thread): race reported
  (the `t1 == t2` relation now has cross-actor pairs).
- A span larger than the stride (`bytes` past `S`): falls back (not decided by this rule).
- A base with no premise bound (wrap possible): falls back.

## Cost

One more exact linearization step in the canonical writer and one relation case in the family
checker; no change to the pairwise path.
