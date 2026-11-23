"""Moves mode end to end on the native executable (tests/moves_cases, docs/moves-mode.md).

ptx/gather.ptx is Coral's k_gather at HID 64 (moves_cases/gen.py says how it was made); every
other PTX file is a hand edit of it with one change, named per test. Inputs and terms are written
by moves_cases/gen.py. Skipped unless SENTRYWARD_PTX_NATIVE names a built sentryward-ptx.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "moves_cases"
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(ptx, entry, term, edit=None):
    """(main report, moves report) of one run; `edit` changes the input JSON first."""
    with tempfile.TemporaryDirectory() as tmp:
        config = HERE / "inputs" / f"{entry}.json"
        if edit:
            data = json.loads(config.read_text())
            edit(data)
            config = Path(tmp) / "input.json"
            config.write_text(json.dumps(data))
        report = Path(tmp) / "moves.json"
        main = subprocess.run([NATIVE, "--ptx", str(HERE / "ptx" / f"{ptx}.ptx"), "--config", str(config),
                               "--moves-term", str(HERE / "terms" / f"{term}.json"),
                               "--moves-report", str(report)],
                              capture_output=True, text=True, check=False, timeout=300)
        return json.loads(main.stdout), json.loads(report.read_text())


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class MovesNativeTest(unittest.TestCase):
    def matched(self, ptx, entry, term):
        main, r = run(ptx, entry, term)
        self.assertEqual(main["result"], "race_free")
        self.assertEqual(r["result"], "matched", r)
        return r

    def refused(self, ptx, entry, term, reason):
        _, r = run(ptx, entry, term)
        self.assertEqual((r["result"], r.get("reason")), ("refused", reason), r)
        return r

    def test_live_case_matches(self):
        r = self.matched("gather", "live", "live")
        self.assertEqual(r["moves"]["writes"], 128)
        self.assertEqual(r["moves"]["moved_bytes"], 256)

    def test_none_case_matches_with_no_write(self):
        r = self.matched("gather", "none", "none")
        self.assertEqual(r["moves"]["writes"], 0)

    # Negative controls: each must be refused.

    def test_term_with_source_row_shifted_refused(self):
        # The term names source unit r + 1; the kernel copies unit r.
        self.assertEqual(self.refused("gather", "live", "live_src_shift", "value")["detail"]["unit"], 0)

    def test_term_with_half_unit_refused(self):
        # The term's unit is half a row: the kernel's second half lies outside it.
        self.refused("gather", "live", "live_half_unit", "unit_bounds")

    def test_kernel_writing_next_destination_row_not_matched(self):
        # gather_dst_next: the store goes to row idx + 1; gather_dst_col: to column i + 1. Memory
        # elimination cannot separate the shifted symbolic rows, so the check is never reached.
        for ptx in ("gather_dst_next", "gather_dst_col"):
            _, r = run(ptx, "live", "live")
            self.assertEqual((r["result"], r["reason"]), ("unsupported", "memory_elimination"), r)
            self.assertEqual(r["detail"]["code"], "symbolic_partial_alias_unsupported")

    def test_kernel_copying_next_source_row_refused(self):
        # gather_src_next: the load reads hn row r + 1 (the move copies the wrong row).
        self.refused("gather_src_next", "live", "live", "value")

    def test_kernel_storing_a_computed_value_refused(self):
        # gather_not: the stored value is ~x, not x.
        self.assertIn("BitNot", self.refused("gather_not", "live", "live", "value")["detail"]["found"])

    def test_write_while_index_is_excluded_refused(self):
        # gather_unguarded: for s < 0 the kernel copies row r to row r instead of skipping it;
        # with every index -1 those writes are refused by the excluded-value guard.
        self.refused("gather_unguarded", "none", "none", "guard")

    def test_live_term_against_none_case_is_no_move(self):
        # live "all" in the none case: the kernel writes nothing, so nothing is matched or moved.
        self.assertEqual(self.matched("gather", "none", "live")["moves"]["moved_bytes"], 0)

    def test_none_term_against_live_case_refused(self):
        self.refused("gather", "live", "none", "guard")


if __name__ == "__main__":
    unittest.main()
