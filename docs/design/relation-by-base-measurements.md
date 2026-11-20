# Relation by base: measurements (k_gdn)

Implementation of [relation-by-base.md](relation-by-base.md) (revision 4) on branch
`claude/sw-relation-by-base`, from the Coral pin 4239eb2. Build: clang 16, Release.

## Result

The rule accepts Coral's H100 `k_gdn` launch for one decode request (decode-b1) without the
pairwise fallback: `race_free` in 575 s (was a 900 s timeout). For two or more requests it cannot
accept, by design: blocks (s, h) and (s', h) of different requests touch the same within-window
offsets, so R holds cross-request pairs that only distinct table entries would separate, and no
premise states that. Those launches decline, as the note requires.

## H100 k_gdn footprint record

Same inputs as the committed record (`bind-sm121/sentryward/h100.json`, input digests equal),
PTX sha256 4ec7e69e, `run.py --jobs 2 --cap 4 --timeout 900 --rows sm121/k_gdn`, 16 GiB per
process. Peak RSS from `/proc/<pid>/status` VmHWM, sampled every 2 s.

| Shape | Before (4239eb2) | After | Wall | Peak RSS |
| --- | --- | --- | --- | --- |
| decode-b1 | timeout 901 s | race_free | 575 s (487 s alone) | 10.0 GiB |
| decode-b2 | timeout 901 s | unsupported: `dynamic_address_overlap_unresolved` | 366 s | 13.5 GiB |
| decode-b4, b8, b16, b24, b32, b40, b48 | timeout 901 s | memory_budget_exceeded | 420 to 868 s | 16 GiB |
| eager-t4096-r48 | timeout 901 s | timeout | 901 s | 15.1 GiB |

decode-b1, traced (uncommitted instrumentation, run alone): TryFamilyMemoryCheck runs twice and
accepts both times, at a flat 2.6 then 4.9 GiB:

| Pass | Families | Pairs related by base | Time |
| --- | --- | --- | --- |
| 1 | 59,812 | 112,525,312 | 29.5 s |
| 2 | 110,752 | 369,098,752 | 57.1 s |

The other ~400 s are outside the memory check. Strategy `conflict-family`, so the fallback is
avoided. Before this change the first pass declined at `family_memory_checker.cpp:442` and the
pairwise fallback took 86% of the time.

decode-b2: the rule declines (cross-request pairs), and the fallback reports the pair it cannot
separate (`src_tab[0] = 0`, `dst_tab[1] = 0`). That is a decided non-pass, not a timeout.

decode-b4: the first family pass accepts in 19 s at 2.6 GiB. The process then exceeds 16 GiB in
the verification stage before the second pass starts. So the overrun is outside this rule. The
older runs timed out before reaching that stage.

## GB10 k_gdn record

`run.py --target gb10 --rows sm121/k_gdn --choice-rows ""` (no branch expansion, as the committed
verdict): unsupported, `control_predicate_not_static` at line 192, 0.5 s. Committed: the same
verdict in 0.7 s. Unchanged: the stop is in the frontend, before any memory check.

## What decides the remaining shapes

- decode-bN, N >= 2: a premise that distinct requests' table entries differ (the monitor's
  injectivity of next/carry entries) would remove the cross-request pairs from R. The note does
  not include it, so the rule declines.
- The 16 GiB overrun after the family pass (b4 and larger) is a separate scale problem in a later
  verification stage.

## Tests

- `tests/test_relation_by_base.py`: 12 controls, all through the whole pipeline. Under clang 16
  all 12 pass. Three mutants each fail the controls: the rule disabled (3 failures), unordered
  pairs of R accepted (4 failures), and the span check dropped (a false `race_free`, 1 failure).
- Full suite under clang 16: 65 tests. The only failures are the six known at 4239eb2
  (test_output_partition x4, test_equivalence_skipped x2).
- GCC 12 does not build `sentryward-ptx` at 4239eb2: 13 frontend translation units fail on nested
  default member initializers (`resources.h:478`), and `source_json.cpp:1817` needs
  `.template get`. The translation units this branch changes in the algorithm library
  (family_memory_checker.cpp, family_access.cpp, symbolic_address_model.cpp) compile under GCC 12.

## Deviations from the note

- Form derivation: the frontend's bounded execution already derives `S * t` exactly. It keeps a
  lookup term through `cvt`, `mul.lo` and `shr` only when the atom's premise range fits the
  widths (`FitsType`), which is the `B * entry_units < 2^31` check per table. So the checker
  re-checks the form instead of adding a second linearization: one lookup atom for every
  member, equal widths, both premise bounds with `lower >= 0`, span within `[0, S)`, and
  `S * upper + span < 2^63` in checked 128-bit arithmetic.
- A family whose members load different table words cannot form: the footprint requires one
  symbolic suffix. Such loads become single-atom families, which the rule decides soundly.
- The shifted-write control ends `unsupported` (unresolved), not `race`. With two distinct table
  atoms, no existing path can prove the overlap.
- An alias-scope control cannot be written as a PTX input. Alias scopes come only from CTA shared
  tensors, which have no lookup atom. `windowForm` and the relation both refuse a nonempty alias.
