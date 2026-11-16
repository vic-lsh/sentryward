"""Footprint verdicts in the public result (docs/output-json.md, Footprint check).

A violation or an undecided access never leaves a race_free result. These tests need no native
executable; the end-to-end cases are in the footprint prototype's run_all.sh.
"""
import unittest

from sentryward.results import RESULTS, is_public, normalize, publish


def native_report(footprint_status):
    return {"status": "race_free", "verification_status": "race_free", "mode": "verify",
            "memory_analysis_complete": True,
            "checks": {"footprint": {"contained": "passed", "violated": "failed",
                                     "inconclusive": "incomplete"}[footprint_status]},
            "footprint": {"status": footprint_status, "violation_count": 1, "undecided_count": 0,
                          "violations": [{"kind": "out_of_bounds"}]}}


class FootprintResultTest(unittest.TestCase):
    def test_category_exists(self):
        self.assertIn("footprint_violation", RESULTS)

    def test_published_report_keeps_the_third_check_and_the_evidence(self):
        out = publish(normalize(native_report("contained"), operation="kernel_verification"))
        self.assertEqual(out["checks"]["footprint"], "passed")
        self.assertIn("footprint", out["evidence"])
        self.assertTrue(is_public(out))

    def test_no_footprint_no_third_check(self):
        report = native_report("contained")
        report.pop("footprint")
        report["checks"] = {}
        out = publish(normalize(report, operation="kernel_verification"))
        self.assertEqual(set(out["checks"]), {"synchronization", "equivalence"})


if __name__ == "__main__":
    unittest.main()
