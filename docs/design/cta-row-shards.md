# Checking a large grid by CTA row shards (k_stage_in/out on H100)

Status: design, for review before code. Owner: Coral SentryWard structural-mode lane.

## Problem

Coral's `k_stage_in` and `k_stage_out` on H100 launch a 512 x 48 grid (24,576 CTAs of 256
threads). Grid row `y` copies 2 MiB between staging row `y` and the state entry
`tab[(req + y) * stride]`. SentryWard executes every CTA (the frontend's per-CTA execution,
`AddCta`, `CtaFacts::PropertyDecisionsSnapshot`) before class admission, and the run exceeds the
16 GiB memory cap at about 290 s (profile, 1 input). The GB10 launch (512 x 8) passes in 293 s.
Class admission does not help: the base of row `y` is a different loaded word, so rows never
join one class, and members are executed before they are admitted.

## Options

A. Cheaper admission (admit a CTA as a translate of its representative without executing it):
   needs a proof that the CTA's control and address shape are the representative's, shifted by a
   known delta, before execution. General but deep (it changes the frontend's unit-major loop).

B. CTA row shards (recommended): check the launch as `K` inputs, shard `k` executing only the
   CTAs with `blockIdx.y` in `[y_k, y_{k+1})`, with the launch's true grid and words (no
   renumbering: addresses are exactly the launch's). Each shard checks containment and race
   freedom among its own CTAs. Cross-shard race freedom is a separate exact check over the
   declared write pieces, plus, where a write address is data-dependent, a premise that is
   checked elsewhere.

## B in detail

- Fork input field `cta_rows: { y: [lo, hi) }`: the frontend executes and lowers only those
  CTAs; footprints and races are over them. Nothing else changes.
- Generator (Coral `coral-mv2-sentryward`): for a launch whose grid exceeds a budget, emit shards
  of whole y rows (GB10-sized, 8 rows each for H100: 6 shards).
- Cross-shard writes:
  - `k_stage_in` writes only staging rows `y` (affine in `y`, declared piece rows): shards write
    disjoint staging rows by the declared pieces alone (checked by `check_verdicts.py` from the
    pieces: row ranges of different shards do not intersect).
  - `k_stage_out` writes state entry `tab[(req + y) * stride]`: shards are disjoint when the
    entries of different rows are distinct. That is the verified monitor's `injective_spec` (a
    step whose two tokens write one state unit is refused; `coral-monitor` `check/injective.rs`,
    control `neg_monitor_injective_state_twice`), so it is a checked premise, not a trusted row;
    the record names it, and `check_verdicts.py` accepts the shard set only with that premise
    named.
  - Reads cannot race with reads; a read in one shard against a write in another is covered by
    the same disjointness (the read and write sets are per-row).
- Coverage: a row is `checked` only when every shard of every launch passes and the cross-shard
  check passes; a missing shard is `trusted` with a reason.

## Soundness obligations (for the reviewer)

1. A shard execution is the launch's execution restricted to its CTAs (same words, grid, inputs).
2. Race freedom of the launch = race freedom within each shard + no conflicting pair across
   shards; the cross-shard condition is decided from declared pieces or the named checked
   premise, never sampled.
3. Containment of the launch = containment of every shard (a union of per-CTA facts).
4. Intra-CTA and CTA-local facts (shared memory, barriers) are unaffected: shards are whole CTAs.

## Controls

- A kernel whose row `y` also writes row `y + 1`'s staging slot: the cross-shard check refuses it.
- `k_stage_out` with the injective premise removed: the shard set is not accepted.
- A shard set missing one row range: coverage stays trusted.

## Cost

A small frontend filter, a generator split and a check in `check_verdicts.py`. Option A stays
open for later.
