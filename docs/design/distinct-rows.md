<!-- Copied from llm-eq branch claude/sw-distinct-entries @ 03a4311de4 (coral/docs/design/sw-distinct-state-entries.md), approved by design review round 2. The SentryWard side is implemented in this fork; see "SentryWard use" and the SentryWard controls. -->

# SentryWard premise: distinct state entries across sequences

Status: design revision 2 (after review 1), for review before code. Consumer: SentryWard's
relation-by-base rule (fork note `relation-by-base.md`, revision 4) on `k_gdn` launches with two
or more sequences.

## Problem

With two or more sequences, `k_gdn` block `s` reads entry `carry_src[s]` and writes entry
`next[s * positions]`, and blocks of different sequences touch the same within-entry offsets, so
relation-by-base declines (H100 decode-b2: `dynamic_address_overlap_unresolved`). The entries'
distinctness must be checked by verified code, not assumed.

## The premise

Per launch, over the `state` format's tables, for sequences `i != j` below `sequences`:

- **W/W:** `next[i * positions] != next[j * positions]`.
- **W/R:** `next[i * positions] != carry_src[j]`.

Same-row pairs (`next[i * positions]` against `carry_src[i]`, equal for an in-place update) and
read/read pairs (fresh sequences share the zero entry; requests at one prefix node share its
snapshot) are not covered and need nothing. Pad rows are sequences too and are covered.

Input format: a SentryWard input key beside `tensor_properties`, fields in the `Premise::field_json`
shape (byte offset, byte stride, count, 64-bit words):

```json
"field_relations": [{"kind": "distinct_rows",
  "write": {"object": "<dst_tab>", "offset": 0, "stride": "<8 * positions>", "count": "<sequences>"},
  "read":  {"object": "<src_tab>", "offset": 0, "stride": 8, "count": "<sequences>"}}]
```

Meaning: for row indices `i != j < count`, `write[i] != write[j]` and `write[i] != read[j]`. The
source is the layout description: `formats/state.toml` gains `[[relation]] kind = "distinct_rows",
write = "next", read = "carry_src"`, and `swpremise` emits it with the same port and launch
evaluation as range premises (strings above are evaluated per shape).

## Discharge

The theorem has two verified halves joined at the composition boundary, plus the existing trust
bridge below. No executable code, no new refusal, zero per-step cost.

**Half 1, words level (coral-monitor).** A conditional property of the checked words, so it holds
in every deployment, with or without a state pool. Open spec `rows_distinct(sin, out)` over two
sequences of unit ids of equal length: for `r != r'`, `out[r] != NULL ==> out[r] != out[r'] &&
out[r] != sin[r']`. Proof fn `Monitor::lemma_accepted_state_rows_distinct(pre, w, live, c)`,
requiring `Monitor::accepted` (owner invariant, planned intent, `checked_spec`), ensures
`rows_distinct` of `sin[r] = view_state_in(r)` and `out[r] = view_state_out(last(r))` for
`r < bucket_rows`, with `last(r) = qo[r + 1] - 1`. Proof:

1. Every row is nonempty: `plan_row_ok` gives `n >= 1` and `row_map_live` gives `seglen == n`;
   `pads_upto` gives `seglen >= 1`. So `last(r)` is a token of row `r`, and `r != r'` gives
   distinct tokens (`shape_spec`: `qo` monotone).
2. W/W: `injective_spec` (two tokens, the first with a non-null state output, have different
   outputs; when the first is null and the second non-null they differ trivially).
3. W/R, by effective holders (`eff`): `eff` only rewrites `Dirty` holders, so on units held `Zero`,
   `Sink`, `Node` or `Req` it is the owner's holder; distinct holders mean distinct units.
   - live output: `eff == Req(q)` (`owned_spec`); pad `d`'s output is sink `sink_base + d`
     (`pad_spec`), holder `Sink(d)` by `reserved_at` in the owner invariant.
   - live input of row `r'` with request `q' != q` (`plan_ok`: live rows name distinct requests,
     which is how packed prefill (D124) lays out rows): the zero unit (holder `Zero`), a unit with
     `eff == Req(q')`, or a node snapshot (`node_snap_in`: holder `Node(x)`).
   - pad input of row `r'`: its own sink `sink_base + d'` with `d' != d`, zero, or null.
   Only rows of the checked step occur; finished or preempted requests have no row in it.

**Half 2, tables level (gate-readers `join_state`).** Proof fn `lemma_entries_distinct`. Requires,
as the reviewer asked, the join, row-map and non-null facts at the boundary, each an existing
verified ensures: `seq_state`'s (for `s < rows`: `seq_ok`, so `next[s * positions] < NULL`;
`state_in[s] == carry_src[s]`; `next_unit[s] == next[s * positions]`); `state_out`'s (token `t`'s
output is `out_of(row, qo, next_unit, t)`); `join::segments`' (`tiles`, `row[t] < rows`,
`in_row`); and `rows_distinct(state_in, out)` with `out[s] = state_out[qo[s + 1] - 1]`. Ensures
`entries_distinct(tabs, positions, rows)`, the premise for all `rows` sequences (the decode rows
`k_gdn` runs are the leading ones). Proof: `in_row` and `tiles` give `row[qo[s + 1] - 1] == s`, so
`out[s] == next[s * positions]`, which is non-null, so Half 1's antecedent holds.
`rows_distinct` is defined once, in coral-monitor; gate-readers imports it (a new edge from
engine framework to lib, the allowed direction; checked by `crate_registry.py strata-check`).

**Composition.** Half 1 is over the words `W` the monitor checked; Half 2 over the vectors
`Staged::read` computed. Their identity (`Staged`'s `view_state_in`, `view_state_out` return those
vectors, and `rows == bucket_rows`) is the plain glue of `Staged::read`, covered by the existing
trusted premise `reader_correct` (obligations ledger) and the monitor-sm121 serve-path row for `Staged`, whose
removal item (move `Staged::read` inside `verus!`) turns the composition into a call. Every check
`checked_spec` passes already rests on that identity, so this adds no trust.

## Trust bridge: checked tables are the launched bytes

The monitor reads host images; `k_gdn` reads device memory. The chain, all existing rows:

| Step | Row |
| --- | --- |
| The tables `read_state` returns are the image bytes | generated `read_state`, verified (D174) |
| The image is what is uploaded, no change after the check | Launcher contract (coral-monitor `launch`, trusted): `Checked` binds the staging buffer, `check_and_launch` checks and launches on one shadow, the Launcher uploads exactly that buffer before the step's kernels |
| The upload lands in the start memory at the table views | `tables_written` (`staged_at`, `views_in_mem`, partial; controls `neg_mv2_tables_outside_upload`, `neg_mv2_tables_unstaged`) |
| Nothing rewrites the tables before `k_gdn` | `writes_checked` (each op writes only checked writes and scratch; staged tables are neither), `reads_checked` (staged words are fixed bytes), `ran` and `ordered` (other streams, events), `dev_host_alias` (an alias is never a write target; GB10 selects `Copy`) |
| `k_gdn`'s table arguments are these tables, read as the format clause | `axiom_row_gdn` argument interpretation (D173) |
| SentryWard's object names are these arguments | tool SentryWard row, "Table premises" clause |

So the reviewer's case (monitor sees `next = [1, 2]`, device holds `[1, 1]` at launch) needs a
write to the staging buffer after the check (excluded by the Launcher binding), an upload other
than the checked one (Launcher), or a step op or alias writing the table (`writes_checked`,
`dev_host_alias`). Range premises rest on rows 2 to 6 too; they do not need `reader_correct`,
because the range check is in the generated reader itself. One row covers both: the "Table
premises" clause of the tool SentryWard row is extended to name relation premises, this note's
lemmas as discharge, and `reader_correct` as the extra premise for relations.

## SentryWard use

Revision 4 relates two families `F1`, `F2` on one resolved allocation, each with form
`c + S t + a(m)`, equal `S`, different atoms. The fork already guarantees one atom per family,
twice: `Footprint` requires every member's symbolic terms equal (else the family is split into
runs of equal suffix), and `Window` rechecks the same term on every member. An atom is a ground
lookup, key `gather:<symbol>(<byte>)`, so each family has one fixed table word. A family spanning
blocks 0 and 1 (words `next[0]` and `next[positions]`) cannot have a window form.

New step: atom `(symbol, byte)` is row `k` of field `f` iff `gather_objects[symbol] == f.object`,
the gather's element width is 8 bytes, `byte >= f.offset`, `(byte - f.offset) % f.stride == 0`
and `k = (byte - f.offset) / f.stride < f.count` (checked 128-bit arithmetic). If a
`distinct_rows` relation makes `t1` write row `i` and `t2` write row `j` or read row `j` with
`i != j` (either order), `t1 == t2` is infeasible and `R` is empty. Any other case (same row, both
read rows, no row, an atom matching no field) keeps revision 4 unchanged.

Obligations:

1. Row derivation exact as above; otherwise no separation.
2. Atom invariance per family: the existing `Footprint` and `Window` checks, kept as preconditions.
3. No store of the launch on either table object (values loaded equal values at launch);
   otherwise the relation is ignored for the launch.
4. Window separation as revision 4 (windows `[S t, S (t + 1))` disjoint for `t1 != t2`).
5. The rule only proves pairs safe; it emits no race and no unresolved verdict.

**decode-b2 qualifies structurally.** Its measured decline is at this rule: the families were
related by base (window forms exist, equal `S`) and `R` held cross-sequence pairs; the fallback
named `src_tab[0] = 0`, `dst_tab[1] = 0`. Its state families have atoms `dst_tab` at bytes `0` and
`8 * positions` and `src_tab` at bytes `0` and `8`: rows 0 and 1 of the relation's fields, so every
cross-sequence pair separates and same-sequence pairs keep the `R` and HB query that decode-b1
passes. Not established: whether b2 then fits 16 GiB in the later stage where b4 and larger
overran; GB10 `k_gdn` stops earlier (`control_predicate_not_static`) and is unaffected.

## Controls

SentryWard, through `TryFamilyMemoryCheck` (fork tests):

- Two-sequence `k_gdn`-shaped kernel with the relation: `race_free`, `base_relations > 0`, no
  fallback, cross-sequence `R` empty. Without it: declines as today.
- Same row (`next[i * p] == carry_src[i]`, in place) with an unordered cross-thread pair in `R`:
  not separated, the fallback reports the race. Mutant separating `i == j`: this control fails.
- Row atom varying within one family: split by `Footprint`; mutant dropping `Window`'s invariance
  check makes a two-row family control fail.
- No separation (decline) for: a 4-byte gather, an offset off the stride, a wrong stride, an index
  at `count`, a `read` field naming `win_off`, a kernel storing to `dst_tab`, two read rows.

Coral:

- monitor v0 image mutations: row 1's `next` set to row 0's `next` (W/W) and to row 0's
  `carry_src` (W/R): refused by `check_step`.
- Proof controls failing `lemma_accepted_state_rows_distinct`: drop `injective_spec`'s state
  clause (W/W); drop `plan_ok`'s distinct requests, or give `node_snap_in` holder `Req` (W/R); drop
  `pads_upto`'s `seglen >= 1`. Failing `lemma_entries_distinct`: `out_of` null on a last token.
- Bridge (existing): Launcher binding (`neg_mv2_tables_outside_upload`, `neg_mv2_tables_unstaged`),
  alias writes (litmus `host_alias`: D2D and H2D writes into an alias refused). A table changed
  after the check: a monitor v0 test that edits the staged image after `Checked` is issued must be
  unable to launch it (new test, Launcher lane).

Generality: stated for any `sequences`, `positions`, `state_units`, pad count, world size and
deployment; no GPU model or bucket size enters the premise, lemmas or format row.

## Model

Not applicable (stateless): no state, transition or protocol changes. Both halves derive facts
from existing specs; the upload and stream ordering they lean on is the existing rows' (model
checker `sound`).

## Alternatives considered

| Design | Guarantees | Proof burden | Openness | Migration |
| --- | --- | --- | --- | --- |
| Two proof lemmas from ownership and the join (chosen) | full; trust as `checked_spec` | two lemmas, no exec | any format | about 1 day |
| Verified run-time check in `seq_state` (sort or bitmap) | full; skips `reader_correct` | small proof, O(rows) per step, new refusal | any format | about 1 day |
| Per-entry objects in SentryWard's address model | full, trusts the split | large SentryWard change | one format | weeks |
| b1 verdict plus a Coral theorem that requests' footprints are disjoint | state only, no bN verdict | Coral-side theorem | any format | days |

The lemmas win: ownership already implies the fact, so a run-time check restates it as a refusal
(the owner prefers static designs), and its trust saving is nil because `reader_correct` already
underlies every monitor verdict. The other two move the fact out of SentryWard's verdict or into
its trust.

## Openness, visibility, parameters

| Set or item | Choice | Reason |
| --- | --- | --- |
| `field_relations` kinds | closed (`distinct_rows`) | SentryWard proves a rule per kind |
| fields a relation names | open (format TOML) | formats own their tables |
| `rows_distinct` | open spec, small, two consumers | interface fact between crates |
| `entries_distinct` | open spec, one consumer (`swpremise` evidence) | the premise's statement |
| `positions`, `sequences` | owner: `state` format params, per launch | `swpremise` launch evaluation |

Files: `rows_distinct` and the Half 1 lemma in a new coral-monitor module `iso/state_rows`, beside
the KV distinctness lemmas; Half 2 in gate-readers `join_state`; the format row in bind-sm121
`formats/state.toml`; the rule in the fork's family checker and symbolic address model.

Design doc: this note.
