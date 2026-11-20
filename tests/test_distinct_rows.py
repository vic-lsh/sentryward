"""Distinct rows (docs/design/distinct-rows.md), end to end on the native executable.

The kernels are tests/relation_by_base_cases/src/rb.cu (see test_relation_by_base.py) launched as two
blocks, one per request: block s reads `state` at window src_tab[s] and writes it at window
dst_tab[s] (variant 6: dst_tab[2 s], k_gdn's next[s * positions]). The inputs are base_grid2.json
plus `field_relations`, written by relation_by_base_cases/make_rows_inputs.py. Skipped unless
SENTRYWARD_PTX_NATIVE names a built sentryward-ptx.

A distinct_rows relation only empties R of a pair related by base whose atoms are rows i != j (write
and write, or write and read). Same-row pairs and two read rows keep revision 4: the rule gives up
and the exact fallback decides, so those launches are not race free here (the tables allow equal
words). The rule never reports a race.
"""
import unittest

from test_relation_by_base import NATIVE, decided_by_base, memory, run


def distinct_row_relations(result):
    return memory(result).get("family_analysis", {}).get("distinct_row_relations", 0)


def code(result):
    return result["diagnostics"][0]["code"]


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class DistinctRowsTest(unittest.TestCase):
    def assertFallback(self, result):
        """Not race free, and not decided by the family checker with distinct rows."""
        self.assertNotEqual(result["result"], "race_free", memory(result))
        self.assertEqual(distinct_row_relations(result), 0, memory(result))
        self.assertFalse(decided_by_base(result), memory(result))

    # Accepted: R of every cross-request pair is empty.

    def test_two_requests_accepted(self):
        r = run("rb0", "rows_grid2")
        m = memory(r)
        self.assertEqual(r["result"], "race_free")
        self.assertTrue(decided_by_base(r), m)
        self.assertGreater(distinct_row_relations(r), 0)
        self.assertEqual(m["overlap_queries"], 0)
        self.assertEqual(m["address_model"]["refined_pairs"], 0)
        # The relation is listed as a trusted premise.
        self.assertEqual(r["frontend_assumptions"]["field_relations"]["relations"][0]["kind"],
                         "distinct_rows")

    def test_positions_stride_accepted(self):
        # next[s * positions], positions = 2: write rows at byte 16 s.
        r = run("rb6", "rows_positions")
        self.assertEqual(r["result"], "race_free")
        self.assertTrue(decided_by_base(r), memory(r))
        self.assertGreater(distinct_row_relations(r), 0)

    def test_without_relation_declines(self):
        for ptx in ("rb0", "rb6"):
            with self.subTest(ptx):
                self.assertFallback(run(ptx, "base_grid2"))

    # Kept as revision 4: the rule gives up, the fallback decides.

    def test_same_row_pair_not_separated(self):
        # Thread t writes a(t) + 4 of window dst_tab[s], which thread t + 1 of the same block reads in
        # window src_tab[s]: rows s and s, equal for an in-place update. That pair of R stays.
        self.assertFallback(run("rb1", "rows_grid2"))

    def test_two_read_rows_not_separated(self):
        # Blocks write windows src_tab[0] and src_tab[1]: both words are rows of `read`, which may
        # be equal (fresh requests share the zero entry).
        self.assertFallback(run("rb8", "rows_grid2"))

    def test_varying_row_atom(self):
        # Thread t reads src_tab[t]: Footprint splits the reads into one family per word, and
        # thread s's read of src_tab[s] in block s' != s meets block s's write at dst_tab[s] (same
        # row): not separated.
        self.assertFallback(run("rb3", "rows_grid2"))

    def test_varying_row_atom_from_a_distinct_row(self):
        # Thread t reads src_tab[(t + 2) & 63]: the first member's word is row 2, distinct from both
        # write rows, but thread 62 reads row 0, block 0's write row. Footprint and windowForm both
        # refuse a family whose members' words differ; a family taking its first member's atom
        # would separate thread 62's same-row pair and report race_free.
        self.assertFallback(run("rb9", "rows_grid2_count64"))

    def test_no_row_declines(self):
        cases = {
            "rows_offset_off_stride": "rb6",  # write rows at 8 + 16 k; the words are at 0 and 16
            "rows_wrong_stride": "rb6",  # stride 24; the word at 16 is no row
            "rows_count1": "rb0",  # the word at 8 is row 1, at count
            "rows_read_other_object": "rb0",  # `read` names dst_tab: src_tab words are no rows
        }
        for inp, ptx in cases.items():
            with self.subTest(inp):
                self.assertFallback(run(ptx, inp))

    # Refused.

    def test_table_written_refused(self):
        # Thread 0 of block s stores to dst_tab[s]: the tables' read-only contract refuses the launch.
        r = run("rb7", "rows_grid2")
        self.assertEqual(r["result"], "invalid_input")
        self.assertEqual(code(r), "constant_input_modified")

    def test_malformed_refused(self):
        cases = {
            "bad_bits": "invalid_field_relation",
            "bad_offset": "invalid_field_relation",
            "bad_stride": "invalid_field_relation",
            "bad_stride_zero": "invalid_value",
            "bad_count_zero": "invalid_value",
            "bad_counts_differ": "invalid_field_relation",
            "bad_out_of_range": "invalid_field_relation",
            "bad_kind": "invalid_field_relation",
            "bad_unknown_object": "unknown_field_relation_object",
            "bad_no_properties": "invalid_field_relation",
            "bad_s32_table": "invalid_field_relation",
        }
        for inp, expected in cases.items():
            with self.subTest(inp):
                r = run("rb0", inp)
                self.assertEqual(r["result"], "invalid_input")
                self.assertEqual(code(r), expected)


if __name__ == "__main__":
    unittest.main()
