"""Reference (real-number) mode end to end on the native executable (tests/value_cases).

The PTX is Coral's k_sact_split (sm_90a, nvcc 12.9 -O3): h = bf16(silu(float(g)) * float(u)) with
silu(x) = x / (1 + expf(-x)). expf(-x) is the compiler's split-log2(e) sequence whose exponent is
-c * x with c = RN32(log2 e)_hi + _lo (the negation folds into the coefficient). Reading exp2(-c * x)
as exp(-x) is the named assumption compiler_constant_transcendentals (an approximation, not real
equality): the verdict lists it with the constant it replaced, and without it the kernel is not
proven. The matcher checks the raw PTX (both split literals, round-to-nearest fmas, one operand):
a kernel without the negation, with an fma.rz, or with either literal changed is not proven, as are
a kernel whose product is an add and a wrong op term (silu(g) + u). The reference is Coral's
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
                           "replaced": "25380159564675/17592186044416", "denotes": "log2(e)", "uses": 1}])
        a = self.idealization(r)
        self.assertEqual((a["enabled"], a["triggered"]), (True, True))

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
