# Moves mode

`--moves-term TERM --moves-report OUT` (`cpp/src/value_analysis/moves.cpp`) checks that one
launch's writes to a destination object are bitwise moves of a source object's units to the units
an index table names. It runs over the memory-eliminated value IR, like structural mode
(`docs/structural-mode.md`), and needs a complete memory elimination, so the synchronization check
must pass first. The term (schema `sentryward.moves-term/v1`) is input data the caller generates
from a `moves` clause of a kernel declaration.

## Term

```json
{"schema": "sentryward.moves-term/v1", "row": "sm121/k_gather", "unit_bytes": 4096,
 "source": {"object": "hn", "base_bytes": 0, "unit_stride_bytes": 4096, "units": 8},
 "destination": {"object": "hs", "base_bytes": 0, "unit_stride_bytes": 4096, "units": 8,
                 "index": {"object": "meta", "element": "s32", "offset_bytes": 44,
                           "stride_bytes": 16, "none": -1}},
 "live": "all"}
```

- Source unit `r` (`r < source.units`) is bytes `[base + r * stride, + unit_bytes)` of the source
  object.
- Its index is the `element` (an integer type) at byte `offset_bytes + r * stride_bytes` of the
  index object. Its destination unit is bytes `[base + index * stride, + unit_bytes)` of the
  destination object.
- `none` is the excluded index value: a source unit whose index is `none` moves nothing.
- `live`: `"all"` when the input's premises exclude `none` for every index (the storing case of a
  case split), `"none"` when they fix every index to `none`. A run whose premises allow both is
  not a moves input: split it by the index field's bounds (one input per interval, as
  `tensor_properties` field bounds), one term per case.

## What a match states

Under the input's premises, for this launch:

1. Every write to the destination object lies in the destination unit of some source unit `r`:
   its position is exactly one symbolic term, the index element of unit `r` times the
   destination's unit stride, plus a constant offset `u` inside the unit.
2. Its value is bytes `[u, u + bytes)` of source unit `r`'s initial contents, bit for bit: the
   stored value is a load (or a constant component of a vector load) resolved to those initial
   bytes, directly or through stored values that are themselves such loads, with only same-width
   bitcasts between. A conversion, any other operation, a reinterpreted or reassembled read, or a
   read at a symbolic position of a written object is refused.
3. No destination byte is written twice.
4. Excluded-value guard: with `live: "none"`, no write to the destination is allowed.

Writes to other objects are counted (`other_writes`) and not checked; the footprint check covers
them. Moves mode does not state that every live destination byte is written (see the coverage
report). Class mode (CTA class representatives only) is `unsupported`: the check reads every CTA's
writes.

## Report

Schema `sentryward.moves-report/v1`. `result`: `matched`, `refused` (the launch does not realize the
term) or `unsupported` (the check cannot decide); neither of the last two is ever a match.
A stop carries `reason`, `message` and `detail` (the write's `offset`, `bytes`, the position's
`index_object`, `index_byte` and `coefficient`, the matched `unit`, the value's `source` region or
the `found` operation).

| `reason` | Kind | Meaning |
| --- | --- | --- |
| `index_map` | refused | a destination write not placed by one index element of a source unit times the unit stride |
| `unit_bounds` | refused | a write outside its destination unit, or a unit of another width |
| `value` | refused | the written bytes are not the same bytes of the source unit (another row, offset, a conversion or a computed value) |
| `partial_read` | refused | the value comes from a reinterpreted or reassembled read |
| `duplicate_write` | refused | one destination byte written twice |
| `guard` | refused | a destination write while every index is the excluded value |
| `memory_elimination` | unsupported | memory elimination did not complete (`detail` names its diagnostic) |
| `class_mode`, `output_form`, `symbolic_read_position`, `load_component`, `unresolved_load`, `thread_selector` | unsupported | a step the check does not decide |
| `value_analysis_not_reached` | unsupported | the run stopped before the value analysis |

A match carries `moves`: `live`, `units`, `unit_bytes`, `writes` (to the destination),
`moved_bytes` and `other_writes` (per allocation).

## Tests

`tests/test_moves_native.py` (`SENTRYWARD_PTX_NATIVE`): Coral's `k_gather` at HID 64 in both cases,
and controls that must not match: a term naming source row `r + 1` or half a unit, a kernel copying
source row `r + 1`, storing `~x`, writing row `idx + 1` or column `i + 1` (memory elimination does
not separate those positions: `unsupported`), and a kernel that writes when the index is `-1`.
`tests/moves_cases/gen.py` writes the inputs and terms.
