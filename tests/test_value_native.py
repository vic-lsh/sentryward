"""Reference (real-number) mode end to end on the native executable (tests/value_cases).

The PTX is Coral's k_sact_split (sm_90a, nvcc 12.9 -O3): h = bf16(silu(float(g)) * float(u)) with
silu(x) = x / (1 + expf(-x)). expf(-x) is the compiler's split-log2(e) sequence whose exponent is
-RN32(log2 e) * x: the negation folds into the coefficient, which the ideal exp rule accepts as
exp(-x). The reference is Coral's generated term of the canonical op silu_mul. Controls: a kernel
with the final product replaced by an add, a kernel with a perturbed log2(e) literal, and a wrong
op term (silu(g) + u) are each not proven. Skipped unless SENTRYWARD_PTX_NATIVE names a built
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


def run(case, ptx):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    d = HERE / case
    out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "sact" / f"{ptx}.ptx"),
                          "--input", str(d / "input.json"), "--reference", str(d / "reference.py"),
                          "--reference-binding", str(d / "reference_binding.json"), "--timeout", "300"],
                         capture_output=True, text=True, env=env, check=False)
    return json.loads(out.stdout)


@unittest.skipUnless(NATIVE and importlib.util.find_spec("torch"), "needs SENTRYWARD_PTX_NATIVE and PyTorch")
class ValueNativeTest(unittest.TestCase):
    def test_negated_compiler_exp_verified(self):
        r = run("sact", "k_sact_split")
        self.assertEqual(r["result"], "verified")
        self.assertEqual(r["internal"]["equivalence"]["counters"]["ideal_compiler_exp_sequences"], 1)

    def test_wrong_kernel_not_proven(self):
        self.assertEqual(run("sact", "k_sact_split_add")["result"], "equivalence_not_proven")

    def test_perturbed_log2e_not_proven(self):
        self.assertEqual(run("sact", "k_sact_split_log2e")["result"], "equivalence_not_proven")

    def test_wrong_term_not_proven(self):
        self.assertEqual(run("sact_wrong_term", "k_sact_split")["result"], "equivalence_not_proven")


if __name__ == "__main__":
    unittest.main()
