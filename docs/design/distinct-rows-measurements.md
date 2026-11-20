# Distinct rows: measurements (k_gdn)

Implementation of the SentryWard side of [distinct-rows.md](distinct-rows.md) on branch
`claude/sw-distinct-rows`, from `claude/sw-relation-by-base` @ 3f612ca. Build: clang 16, Release.

## Result

The relation does what the note expects in the memory check: on H100 `k_gdn` decode-b2, the
family checker now accepts both passes without the pairwise fallback, and 1.0 G of the
1.7 G family pairs related by base in pass 2 are separated by distinct rows. The launch is still
not decided in the 900 s box. Pass 2 is quadratic in families, and it ends at about 860 s,
too late for the later stages. decode-b4 still exceeds 16 GiB. decode-b1 is unchanged
(race_free; its pairs are all same-request).

## H100 k_gdn record

Inputs: `sw_inputs` from llm-eq origin/main 8596a0718c (includes #5501) for
`qwen36-h100-mvp.toml`, with `field_relations` added to every `k_gdn` line by the script in the
appendix (the Coral generator does not emit it yet). PTX sha256 4ec7e69e (recurrent.cu, nvcc 12.9,
`nvcr.io/nvidia/pytorch:25.04-py3`). `run.py --jobs 2 --cap 4 --timeout 900 --rows sm121/k_gdn
--native <this branch @ 645de2a>`, 16 GiB per process. Peak RSS is VmHWM of `/proc/<pid>/status`,
sampled every 2 s (a process killed at the cap is reported at its last sample).

| Shape | Input digest | relation-by-base (3f612ca, no relation) | This branch | Wall | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| decode-b1 | 88bed66c | race_free, 575 s | race_free | 652 s | 10.0 GiB |
| decode-b2 | 4a986e8a | unsupported: `dynamic_address_overlap_unresolved`, 366 s | timeout | 903 s | 13.5 GiB |
| decode-b4 | eff6cd99 | memory_budget_exceeded | memory_budget_exceeded | 406 s | >= 14.6 GiB (killed at 16) |

decode-b8 was not run, because b4 does not fit. Host load was about 100 during all runs.

## decode-b2, traced

The trace was an uncommitted stderr line at the `TryFamilyMemoryCheck` call, from a run alone
with 900 s and 16 GiB.

| Build | Pass 1 | Pass 2 |
| --- | --- | --- |
| 7306058 (R computed, then emptied) | accepts in 50 s: 59,812 families, 112,525,312 pairs by base, 0 by distinct rows | starts at 401 s, still running at 900 s |
| 645de2a (distinct rows tested before R) | accepts in 44 s, same counts | starts at 324 s, accepts in 534 s: 221,504 families, 1,744,830,464 pairs by base, 1,006,632,960 by distinct rows; no fallback |

In the 645de2a run, pass 2 ended 858 s after the start, and the stages after it did not finish by
900 s. Peak RSS was 15.6 GiB. The rule itself needs no HB request for a separated pair. The
remaining cost is enumerating 1.7 G family pairs: the candidate loop visits every write family
against every family on the allocation.

## What decides the remaining shapes

- decode-b2 needs pass 2 to be sub-quadratic in families, for example grouping families by
  table word so that separated groups are skipped without visiting their pairs. That change is
  outside the note.
- decode-b4 and larger still overrun 16 GiB in a later verification stage, as at 3f612ca.

## Tests

- `tests/test_distinct_rows.py`: 10 controls through the whole pipeline, all passing. Mutants
  (each built and run): same-row pairs separated gives a false race_free on the same-row
  control; read/read rows separated gives a false race_free on the two-read-rows control;
  distinctRows disabled makes both accept tests fail. Dropping Window's per-member atom check
  alone, or Footprint's alone, is caught by the other; dropping both gives a false race_free on
  `test_varying_row_atom_from_a_distinct_row`.
- `tests/test_relation_by_base.py`: 12/12. Full suite (clang 16): 75 tests; the only failures are the six known at 4239eb2 (test_output_partition x4, test_equivalence_skipped x2). Under host load, the tests that expect a particular
  fallback diagnostic or a fallback race_free sometimes see `address_solver_inconclusive`, because
  the SMT budget is 100 ms per pair.
- GCC 12 still does not build `sentryward-ptx`, for the reasons recorded at 3f612ca
  (`source_json.cpp:1817`, nested default member initializers reaching `run.cpp` and 12 other
  frontend units). The translation units this branch changes that GCC 12 reaches compile:
  family_memory_checker.cpp, symbolic_address_model.cpp, public_input_parse.cpp and
  public_input_normalize.cpp.

## Deviations from the note

- A 4-byte gather is refused at input (the relation's objects must have 64-bit
  `tensor_properties`) instead of declined.
- A kernel storing to a table is refused by the existing tensor_properties read-only contract
  (`constant_input_modified`) instead of the relation being ignored for the launch. Requiring
  `tensor_properties` on both objects makes this the only case.
- `bits` is optional and must be 64. Both fields must have one `count`.

## Appendix: adding field_relations to generated inputs

Each `k_gdn` line gets, from its own range-premise fields (`src_tab`: offset 0, stride 8,
count `sequences`; `dst_tab`: offset 0, stride 8, count `sequences * positions`):

```python
positions = d["count"] // s["count"]            # d, s: the dst_tab and src_tab fields
inp["field_relations"] = [{"kind": "distinct_rows",
    "write": {"object": "dst_tab", "offset": 0, "stride": 8 * positions, "count": s["count"], "bits": 64},
    "read":  {"object": "src_tab", "offset": 0, "stride": 8, "count": s["count"], "bits": 64}}]
line = json.dumps(rec, separators=(",", ":"))     # `input` stays the last key, as run.py reads it
```

Other lines are copied byte for byte. Each edited line was checked to equal its original after
removing `field_relations`. Decode shapes have `positions = 1`.
