"""Reference (real-number) mode end to end on the native executable (tests/value_cases).

The PTX is Coral's k_sact_split (sm_90a, nvcc 12.9 -O3): h = bf16(silu(float(g)) * float(u)) with
silu(x) = x / (1 + expf(-x)). expf(-x) is the compiler's split-log2(e) sequence whose exponent is
-c * x with c = RN32(log2 e)_hi + _lo (the negation folds into the coefficient). Reading exp2(-c * x)
as exp(-x) is the named assumption compiler_constant_transcendentals (an approximation, not real
equality): the verdict lists it with the constant it replaced, and without it the kernel is not
proven. The matcher checks the raw PTX (both split literals, round-to-nearest fmas, one operand; on
sm_121a nvcc negates both literals instead of the operand): a kernel without the negation, with an
fma.rz, with either literal changed or with mixed literal signs is not proven, as are
a kernel whose product is an add and a wrong op term (silu(g) + u). Provenance is per raw occurrence:
a second sequence with the same canonical exponent but swapped literals inherits nothing. The reference is Coral's
generated term of the canonical op silu_mul. Skipped unless SENTRYWARD_PTX_NATIVE names a built
sentryward-ptx and PyTorch is installed.
"""
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "value_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(case, ptx, *extra):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    d = HERE / case
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "sact" / f"{ptx}.ptx"),
                          "--input", str(d / "input.json"), "--reference", str(d / "reference.py"),
                          "--reference-binding", str(d / "reference_binding.json"), "--timeout", "300", *extra],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


@unittest.skipUnless(NATIVE and importlib.util.find_spec("torch"), "needs SENTRYWARD_PTX_NATIVE and PyTorch")
class ValueNativeTest(unittest.TestCase):
    def idealization(self, r):
        return next(a for a in r["assumptions"] if a["key"] == "compiler_constant_transcendentals")

    def test_negated_compiler_exp_verified_under_named_assumption(self):
        r = run("sact", "k_sact_split")
        self.assertEqual(r["result"], "verified")
        self.assertEqual(r["internal"]["equivalence"]["idealizations"],
                         [{"assumption": "compiler_constant_transcendentals", "rule": "split_log2e_exp",
                           "replaced": "12102203/8388608 + 338819/17592186044416 = 25380159564675/17592186044416",
                           "denotes": "log2(e)", "uses": 1}])
        a = self.idealization(r)
        self.assertEqual((a["enabled"], a["triggered"]), (True, True))

    def test_assumption_triggered_with_range_premise_disabled(self):
        # The two comparison-derived assumptions are reported independently.
        r = run("sact", "k_sact_split", "--no-assume", "floating_values_in_range")
        if r["result"] == "verified":
            a = self.idealization(r)
            self.assertEqual((a["enabled"], a["triggered"]), (True, True))
        else:
            self.assertTrue(r["internal"]["equivalence"]["idealizations"])

    def test_two_checked_occurrences_verified(self):
        # Positive control for the shadow shape: a second, checked split sequence on the same operand and
        # a cancelling add/sub of the first one's product.
        self.assertEqual(run("sact", "k_sact_split_shadow_ok")["result"], "verified")

    def test_witness_does_not_transfer_between_occurrences(self):
        # The second sequence swaps the literals (L_hi outer): its canonical exponent equals the checked
        # first sequence's, but provenance is per raw occurrence, so only the first product is rewritten.
        r = run("sact", "k_sact_split_shadow")
        self.assertEqual(r["result"], "equivalence_not_proven")
        self.assertEqual([(i["rule"], i["uses"]) for i in r["internal"]["equivalence"]["idealizations"]],
                         [("split_log2e_exp", 1)])

    def test_implicit_approximate_division_listed(self):
        # PTX ISA 1.3: div.f32 with no rounding modifier is approximate, so it is named too.
        r = run("sact", "k_sact_split_implicit_div")
        self.assertEqual(r["result"], "verified")
        self.assertEqual(sorted(r["internal"]["equivalence"]["approximate_instructions"]),
                         ["div.f32", "ex2.approx.ftz.f32"])

    def test_sm121a_folded_negation_verified(self):
        # sm_121a: nvcc folds the negation into both literals (-L_hi, -L_lo), no neg.f32.
        r = run("sact", "k_sact_split_sm121a")
        self.assertEqual(r["result"], "verified")
        # The replaced constants are the negated literals: reported with their sign.
        self.assertEqual(r["internal"]["equivalence"]["idealizations"],
                         [{"assumption": "compiler_constant_transcendentals", "rule": "split_log2e_exp",
                           "replaced": "-12102203/8388608 - 338819/17592186044416 = -25380159564675/17592186044416",
                           "denotes": "-log2(e)", "uses": 1}])

    def test_approximate_instructions_listed(self):
        # ex2.approx (read as exp2) is named by its PTX mnemonic, apart from the idealizations; the kernel's
        # div.rn (exactly rounded) is not an approximate instruction.
        r = run("sact", "k_sact_split")
        approximate = r["internal"]["equivalence"]["approximate_instructions"]
        self.assertEqual(sorted(approximate), ["ex2.approx.ftz.f32"])
        self.assertGreater(approximate["ex2.approx.ftz.f32"], 0)

    # Configure with -DSENTRYWARD_TIMEOUT_TEST_HOOK=ON and set the same-named environment variable.
    @unittest.skipUnless(os.environ.get("SENTRYWARD_TIMEOUT_TEST_HOOK"), "needs test-hook build")
    def test_timeout_reports_approximate_instructions(self):
        from unittest.mock import patch
        from sentryward.frontend.ptx_native import _run_ptx
        d = HERE / "sact"
        captured = subprocess.run(
            [sys.executable, "-m", "sentryward._capture_reference", str(d / "reference.py"),
             str(d / "input.json"), str(d / "reference_binding.json")],
            capture_output=True, check=True, timeout=120)
        with patch.dict(os.environ, {"SENTRYWARD_TEST_HOLD_COMPARISON": "1"}):
            r = _run_ptx(d / "k_sact_split_implicit_div.ptx", d / "input.json",
                         reference=captured.stdout, reference_binding=d / "reference_binding.json",
                         pipeline_timeout_seconds=20)
        self.assertEqual(r["result"], "timeout")
        self.assertEqual(r["internal"]["equivalence"]["stage"], "equivalence.object_comparison")
        counts = r["internal"]["equivalence"]["approximate_instructions"]
        self.assertEqual(sorted(counts), ["div.f32", "ex2.approx.ftz.f32"])
        self.assertEqual(counts, {"div.f32": 512, "ex2.approx.ftz.f32": 512})

    def test_mixed_literal_signs_not_proven(self):
        r = run("sact", "k_sact_split_sm121a_mixed_sign")
        self.assertEqual(r["result"], "equivalence_not_proven")
        self.assertEqual(r["internal"]["equivalence"]["idealizations"], [])

    def test_without_the_assumption_not_proven(self):
        r = run("sact", "k_sact_split", "--no-assume", "compiler_constant_transcendentals")
        self.assertEqual(r["result"], "equivalence_not_proven")
        self.assertEqual(r["internal"]["equivalence"]["idealizations"], [])

    def test_missing_negation_not_proven(self):
        self.assertEqual(run("sact", "k_sact_split_no_neg")["result"], "equivalence_not_proven")

    def test_other_rounding_not_proven(self):
        # fma.rz for the L_lo fma: same canonical coefficient, refused by the raw matcher.
        r = run("sact", "k_sact_split_rz")
        self.assertEqual(r["result"], "equivalence_not_proven")
        self.assertEqual(r["internal"]["equivalence"]["idealizations"], [])

    def test_other_constant_not_proven(self):
        for ptx in ("k_sact_split_log2e_hi", "k_sact_split_log2e_lo"):
            self.assertEqual(run("sact", ptx)["result"], "equivalence_not_proven", ptx)

    def test_wrong_kernel_not_proven(self):
        self.assertEqual(run("sact", "k_sact_split_add")["result"], "equivalence_not_proven")

    def test_wrong_term_not_proven(self):
        self.assertEqual(run("sact_wrong_term", "k_sact_split")["result"], "equivalence_not_proven")

if __name__ == "__main__":
    unittest.main()
