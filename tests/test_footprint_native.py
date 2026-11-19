"""Footprint check end to end on the native executable (tests/footprint_cases).

The PTX is Coral's sm_121a k_add_norm, k_attn_gate and k_gather (nvcc 12.9.86 -O3) and planted
mutants of it. Skipped unless SENTRYWARD_PTX_NATIVE names a built sentryward-ptx.

Covers: CTA class mode with footprints (a 1024-row launch is checked through class representatives
moved by each member's delta, and a write past the last row is still refused), the Coral read rule
(`footprint_reads: noninterference`: a read outside the declared bytes is reported, not refused;
writes keep containment), and declared pieces for symbolic offsets whose coefficients are
multiples of the row stride (a row index read from a table).
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "footprint_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(ptx, inp, *extra):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "ptx" / f"{ptx}.ptx"),
                          "--input", str(HERE / "inputs" / f"{inp}.json"), "--timeout", "300", *extra],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class FootprintNativeTest(unittest.TestCase):
    def footprint(self, result):
        return result["evidence"]["footprint"]

    def test_class_mode_large_launch_contained(self):
        r = run("rowwise", "add_norm_1024")
        fp = self.footprint(r)
        self.assertEqual(r["result"], "race_free")
        self.assertEqual((fp["writes"], fp["reads"]), ("contained", "contained"))
        # Representatives only: far fewer analysed accesses than the 512 x 12288 of the launch.
        self.assertLess(fp["checked_accesses"], 1024 * 12288 // 8)
        hn = next(o for o in fp["objects"] if o["object"] == "hn")
        self.assertEqual((hn["writes"], hn["inferred_max_byte_exclusive"]), (1024 * 2048, 1024 * 4096))

    def test_class_mode_write_past_last_row_refused(self):
        r = run("rowwise_oob_write", "add_norm_1024")
        fp = self.footprint(r)
        self.assertEqual(fp["writes"], "violated")
        self.assertIn("offset=%d" % (1024 * 4096), [v.get("witness") for v in fp["violations"]])

    def test_noninterference_read_outside_is_reported_not_refused(self):
        r = run("rowwise_oob_read", "add_norm_noninterference")
        fp = self.footprint(r)
        self.assertEqual(r["checks"]["footprint"], "passed")
        self.assertEqual((fp["read_rule"], fp["writes"], fp["reads"]), ("noninterference", "contained", "not_contained"))
        self.assertGreater(fp["reads_outside_declared"], 0)
        self.assertEqual(fp["outside_reads"][0]["object"], "w")

    def test_noninterference_keeps_write_containment(self):
        fp = self.footprint(run("rowwise_oob_write", "add_norm_noninterference"))
        self.assertEqual(fp["writes"], "violated")

    def test_symbolic_rows_inside_pieces(self):
        fp = self.footprint(run("gather", "gather_rows_periodic"))
        self.assertEqual(fp["writes"], "contained")
        self.assertEqual(fp["undecided_count"], 0)

    def test_symbolic_rows_outside_pieces_refused(self):
        fp = self.footprint(run("gather", "gather_rows_half_width"))
        self.assertEqual(fp["writes"], "violated")
        self.assertEqual({v["kind"] for v in fp["violations"]}, {"outside_declared_pieces"})

    def test_constant_pieces_still_checked(self):
        fp = self.footprint(run("rowwise_attn_wrong_half", "attn_gate_periodic"))
        self.assertEqual(fp["status"], "violated")

    def test_piece_sets_contained(self):
        fp = self.footprint(run("rowwise", "add_norm_pieces"))
        self.assertEqual((fp["writes"], fp["reads"], fp["undecided_count"]), ("contained", "contained", 0))

    def test_piece_set_short_of_the_last_row_refused(self):
        fp = self.footprint(run("rowwise", "add_norm_pieces_short"))
        self.assertEqual(fp["writes"], "violated")
        self.assertEqual({(v["kind"], v["object"]) for v in fp["violations"]}, {("outside_declared_pieces", "hn")})

    def test_piece_direction_refuses_a_write(self):
        fp = self.footprint(run("rowwise", "add_norm_pieces_x_read"))
        self.assertEqual(fp["writes"], "violated")
        self.assertIn("write_to_read_only", {v["kind"] for v in fp["violations"]})


if __name__ == "__main__":
    unittest.main()
