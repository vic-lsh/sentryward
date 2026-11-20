"""Strided access families in canonical address mode, end to end on the native executable.

The PTX (tests/strided_cases/ptx/strided.ptx) is built from tests/strided_cases/src/strided.cu
(nvcc 12.9, -O3 -std=c++17 --ptx -arch=sm_90a). Skipped unless SENTRYWARD_PTX_NATIVE names a
built sentryward-ptx.

A grid-stride copy whose base is loaded from a table word the launch never writes (Coral's
k_stage_in/out, k_gdn) used to give one access family per thread-iteration: every Load node of the
canonical address DAG carried its per-thread occurrence and every address was one opaque atom.
Now a load of an unwritten allocation is keyed by its address alone, and the address's
coordinate-constant summands are peeled into the affine constant, so a family's members share
their atoms and differ by a constant step. The table word has no tensor_properties premise here,
so it stays an opaque load (with a premise it is a gather atom and merged before this change).

Family counts below are internal.memory_analysis.family_analysis.access_families. Without the
change: copy_ro 2088, copy_once 528, copy_broken_stride 534 (the negatives are unchanged).
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "strided_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(inp):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "ptx" / "strided.ptx"),
                          "--input", str(HERE / "inputs" / f"{inp}.json"), "--timeout", "300"],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


def families(result):
    return result["internal"]["memory_analysis"]["family_analysis"]["access_families"]


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class StridedFamiliesNativeTest(unittest.TestCase):
    # 4x2 CTAs of 64 threads, 4 iterations each; the PTX has 11 global accesses (base load, 5 copies).
    def test_read_only_base_merges_and_keeps_verdict(self):
        r = run("copy_ro")
        fp = r["evidence"]["footprint"]
        self.assertEqual(r["result"], "race_free")
        self.assertEqual((r["checks"]["footprint"], fp["writes"]), ("passed", "contained"))
        self.assertLessEqual(families(r), 8 * 11)  # 72 now; 2088 before: one per thread-iteration

    def test_written_base_keeps_per_thread_atoms(self):
        # Thread 0 of each CTA row writes a tab word: the base load keeps its occurrence.
        r = run("copy_written_base")
        self.assertEqual(r["result"], "race_free")
        self.assertGreaterEqual(families(r), 2048)

    def test_per_thread_table_base_does_not_merge_and_overlap_is_reported(self):
        # Thread t writes out[tab[t]]: its own word, a different atom per thread (2 x 64 threads).
        r = run("table_scatter")
        self.assertGreaterEqual(families(r), 128)
        self.assertNotEqual(r["result"], "race_free")
        codes = [d.get("code") for d in r.get("diagnostics", [])]
        self.assertIn("dynamic_address_overlap_unresolved", codes)

    def test_unbroken_copy_is_one_family_per_instruction(self):
        r = run("copy_once")
        self.assertEqual(r["result"], "race_free")
        self.assertEqual(families(r), 24)  # 528 before

    def test_broken_stride_splits_family_and_race_is_reported(self):
        # Thread 37 writes element 38, as thread 38 does: the store family splits at the break.
        r = run("copy_broken_stride")
        self.assertEqual(r["result"], "potential_race")
        self.assertGreater(families(r), 24)
        race = r["evidence"]["races"][0]
        self.assertEqual(race["region"]["allocation"], "staging")
        self.assertIn(race["region"]["offset"], (38 * 16, 16384 + 38 * 16))


if __name__ == "__main__":
    unittest.main()
