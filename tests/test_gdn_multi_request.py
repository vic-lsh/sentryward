"""k_gdn with two or more requests (docs/design/gdn-multi-request.md), end to end on the native
executable. Skipped unless SENTRYWARD_PTX_NATIVE names a built sentryward-ptx.

A2 (footprint): class mode keeps the representative's symbolic containment verdict for its class
members only when the verdict's basis is the policy hull and every member's translated lookup
element lies in exactly the field policies of the representative's element. The reviewer's launch
(tests/gdn_multi_cases/ptx/rev.ptx): two one-warp CTAs, thread 0 reads 4 bytes of
state[tab[blockIdx.x]], state is 4 bytes, tab elements in [0, 1] and an address invariant
tab[0] = 0. The representative (CTA 0) is contained only by the solver (from tab[0] = 0); CTA 1
may read [4, 8). Class mode must restart and the per-unit run must report CTA 1.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "gdn_multi_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(ptx, inp):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "ptx" / f"{ptx}.ptx"),
                          "--input", str(HERE / "inputs" / f"{inp}.json"), "--timeout", "300"],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


def restart(result):
    return (result.get("internal", {}).get("frontend", {}).get("unit_restart") or {}).get("reason")


def footprint(result):
    return result.get("evidence", {}).get("footprint", {})


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class ContainmentBasisTest(unittest.TestCase):
    def assertReportsCta1(self, result):
        fp = footprint(result)
        self.assertEqual(result["result"], "footprint_violation", (result["result"], fp.get("status")))
        state = [v for v in fp["violations"] if v["object"] == "state"]
        self.assertEqual([(v["cta"], v["side"]) for v in state], [(1, "past_extent")], fp)

    def test_pinned_field_policy_restarts_and_reports_cta1(self):
        # Field policy 0 <= tab[i] <= 1 (i < 2), tab[0] = 0 pinned: the basis is the solver.
        r = run("rev", "rev_field")
        self.assertReportsCta1(r)
        self.assertEqual(restart(r), "footprint_symbolic_member:state:basis_solver")

    def test_pinned_object_policy_restarts_and_reports_cta1(self):
        # The same bound as an object-wide policy: not field bounded, still decided by the solver.
        # The check before A2 restarted only for a field-bounded atom and transferred this verdict.
        r = run("rev", "rev_object")
        self.assertReportsCta1(r)
        self.assertEqual(restart(r), "footprint_symbolic_member:state:basis_solver")

    def test_unpinned_reports_both_ctas(self):
        for inp in ("rev_field_nopin", "rev_object_nopin"):
            with self.subTest(inp):
                fp = footprint(run("rev", inp))
                state = sorted(v["cta"] for v in fp["violations"] if v["object"] == "state")
                self.assertEqual(state, [0, 1], fp)

    def test_policy_hull_transfers_in_class_mode(self):
        # state 8 bytes: the hull [0, 8) fits, so the basis is the policy hull; CTA 1's element 4 is
        # in the same field policies as element 0 (or the bound is object-wide). Class mode keeps
        # the verdict: no restart, one analysed state read for both CTAs.
        for inp in ("hull_field", "hull_object"):
            with self.subTest(inp):
                r = run("rev", inp)
                fp = footprint(r)
                self.assertEqual(r["result"], "race_free", fp)
                self.assertIsNone(restart(r))
                self.assertEqual(fp["verdict_basis"]["policy_hull"], 1, fp)
                self.assertEqual(fp["checked_accesses"], 3, fp)

    def test_differing_field_policy_sets_restart(self):
        # tab[0] <= 1, tab[1] <= 2^30 (two one-row field policies): the representative's hull fits,
        # CTA 1's element lies in another policy. Restart; the per-unit run reports CTA 1.
        r = run("rev", "rows_differ")
        self.assertReportsCta1(r)
        self.assertEqual(restart(r), "footprint_symbolic_member:state:policy_set")


if __name__ == "__main__":
    unittest.main()
