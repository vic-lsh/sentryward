"""Relation by base (docs/design/relation-by-base.md), end to end on the native executable.

The PTX is tests/relation_by_base_cases/src/rb.cu (nvcc 12.9 -O3 -arch=sm_90a, one PTX per
VARIANT): a k_gdn-shaped kernel whose block reads `state` at window src_tab[s] and writes it at
window dst_tab[s], a window being 256 * entry_units bytes. Inputs: entry_units = 4 (S = 1024 bytes),
src_tab in [0, 3], dst_tab in [0, 7] unless the input says otherwise. Skipped unless SENTRYWARD_PTX_NATIVE
names a built sentryward-ptx.

Every case runs the whole pipeline, so the family checker (TryFamilyMemoryCheck) is exercised as
deployed. `decided_by_base`: the family checker answered (strategy conflict-family) with at least
one pair related by base; a declined launch reports the fallback's strategy and no family
statistics. The rule only proves safety: a race or an unresolved verdict comes from existing code.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "relation_by_base_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(ptx, inp):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "ptx" / f"{ptx}.ptx"),
                          "--input", str(HERE / "inputs" / f"{inp}.json"), "--timeout", "300"],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


def find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find(value, key)
            if found is not None:
                return found
    return None


def memory(result):
    return find(result, "memory_analysis") or {}


def decided_by_base(result):
    m = memory(result)
    return m.get("strategy") == "conflict-family" and m.get("family_analysis", {}).get("base_relations", 0) > 0


def base_relations(result):
    return memory(result).get("family_analysis", {}).get("base_relations", 0)


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class RelationByBaseTest(unittest.TestCase):
    def assertFallback(self, result):
        """The rule did not decide the launch: the family checker declined (another strategy) or
        no family pair was related by base."""
        self.assertFalse(decided_by_base(result), memory(result))

    # Accepted.

    def test_same_map_accepted_without_fallback(self):
        r = run("rb0", "base")
        m = memory(r)
        self.assertEqual(r["result"], "race_free")
        self.assertTrue(decided_by_base(r), m)
        # The state families never reach the address model's pairwise query.
        self.assertEqual(m["overlap_queries"], 0)
        self.assertEqual(m["address_model"]["refined_pairs"], 0)

    def test_per_head_offsets_two_ctas_accepted(self):
        # k_gdn's decode-b1 shape: one request, blocks differ by a per-head offset (rec_off).
        r = run("rb5", "base_heads2")
        self.assertEqual(r["result"], "race_free")
        self.assertTrue(decided_by_base(r), memory(r))
        self.assertEqual(memory(r)["overlap_queries"], 0)

    def test_varying_atom_split_into_single_atom_families(self):
        # Thread t reads src_tab[t]: family formation already refuses a family whose members carry
        # different atoms (equal symbolic suffix), so each such read is its own family with one
        # atom, and the rule decides those soundly (thread t's bytes 4t of any window meet only
        # thread t's write). No family with a varying atom reaches the rule.
        r = run("rb3", "base")
        self.assertEqual(r["result"], "race_free")
        self.assertTrue(decided_by_base(r), memory(r))

    # The rule gives up; the exact fallback decides.

    def test_two_requests_decline(self):
        # Two blocks (requests) with the same within-window offsets: R holds cross-block pairs that
        # only distinct table entries would separate, which no premise states. The rule gives up;
        # the fallback reports the pair it cannot separate.
        r = run("rb0", "base_grid2")
        self.assertFallback(r)
        self.assertEqual(r["result"], "unsupported")
        self.assertEqual(r["diagnostics"][0]["code"], "dynamic_address_overlap_unresolved")

    def test_shifted_write_declines(self):
        # Thread t writes a(t) + 4, the bytes thread t + 1 reads when both tables name one entry:
        # an unordered cross-thread pair of R. The rule gives up; the fallback cannot decide
        # whether src_tab[0] == dst_tab[0], so the launch is not race free.
        r = run("rb1", "base")
        self.assertFallback(r)
        self.assertEqual(r["result"], "unsupported")
        self.assertEqual(r["diagnostics"][0]["code"], "dynamic_address_overlap_unresolved")

    def test_disjoint_windows_cross_actor_pair_declines(self):
        # Premises force t1 in [0, 1] and t2 in [2, 3] (single values would fold to constants): R
        # still holds the cross-thread pair, so the rule gives up (it never uses t1 != t2); the
        # fallback proves the windows disjoint.
        r = run("rb1", "disjoint_windows")
        self.assertFallback(r)
        self.assertEqual(r["result"], "race_free")

    def test_unequal_widths_decline(self):
        # 8-byte reads at 8t against 4-byte writes at 4t: byte ranges of different threads meet.
        r = run("rb4", "base")
        self.assertFallback(r)
        self.assertEqual(r["result"], "unsupported")

    # No window form: the existing path, unchanged.

    def test_span_past_stride_falls_back(self):
        # S = 256 bytes; 128 floats span 512.
        r = run("rb0", "span_past_stride")
        self.assertEqual(base_relations(r), 0)
        self.assertFallback(r)

    def test_inexact_truncation_falls_back(self):
        # tab <= 2 with entry_units = 2^30: tab * entry_units reaches 2^31, so (int32_t) is not
        # the identity and no lookup form is derived.
        r = run("rb0", "truncation_inexact")
        self.assertEqual(base_relations(r), 0)
        self.assertFallback(r)

    def test_loose_destination_bound_falls_back(self):
        # Source tight (state_units = 1), destination loose (next < 3), entry_units = 2^30: the
        # destination family has no form.
        r = run("rb0", "loose_destination")
        self.assertEqual(base_relations(r), 0)
        self.assertFallback(r)

    def test_missing_premise_falls_back(self):
        for inp in ("no_premise", "no_upper_bound"):
            with self.subTest(inp):
                r = run("rb0", inp)
                self.assertEqual(base_relations(r), 0)
                self.assertFallback(r)

    def test_same_word_race_reported_by_existing_path(self):
        # Two threads write one float of the same window (one table word, one atom): the existing
        # same-suffix relation reports the race; nothing is related by base.
        r = run("rb2", "base")
        self.assertEqual(r["result"], "potential_race")
        self.assertEqual(base_relations(r), 0)


if __name__ == "__main__":
    unittest.main()
