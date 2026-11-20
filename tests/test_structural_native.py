"""Structural mode end to end on the native executable (tests/structural_cases).

The PTX is tests/structural_cases/src/roles.cu (nvcc 12.9 -O3 --ptx, sm_90a) and the hand-written
ptx/handwritten.ptx (cases nvcc would fold away). Skipped unless SENTRYWARD_PTX_NATIVE names a
built sentryward-ptx.

Covers the role boundary (a body argument's transcendental role is checked from its producer's
bound output; an input argument's from its loads, over every path including through a producer)
producer binding (an unstored inner node's element, identified by its coordinate, binds one value
at every use; an unstored primitive producer is refused, its operation would go unchecked) and the
Multiplicand role (no path from the argument bypasses the multiply).
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "structural_cases"
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")


def run(entry, term, ptx="roles"):
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run([NATIVE, "--ptx", str(HERE / "ptx" / f"{ptx}.ptx"), "--config",
                        str(HERE / "inputs" / f"{entry}.json"), "--analyze-scalar-fold",
                        "--structural-term", str(HERE / "terms" / f"{term}.json"),
                        "--structural-report", str(report)],
                       capture_output=True, text=True, check=False, timeout=300)
        return json.loads(report.read_text())


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class StructuralNativeTest(unittest.TestCase):
    def test_shared_input_without_transcendental_matches(self):
        self.assertEqual(run("shared_pos", "shared")["result"], "matched")

    def test_shared_input_through_a_producer_sigmoid_refused(self):
        # C(P(a), a), P = sigmoid: a reaches C's output through the producer's transcendental, so
        # the avoids role on input a fails even though the producer's output is a cut value.
        r = run("shared_neg_sigmoid", "shared")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role"))
        self.assertEqual(r["detail"]["role"], "avoids_transcendental")

    def test_unstored_producer_one_value_matches(self):
        self.assertEqual(run("unstored_pos", "unstored")["result"], "matched")

    def test_unstored_producer_two_values_refused(self):
        # Two values with q's leaves at q's two uses: one inner node element, two bindings.
        r = run("unstored_neg_two", "unstored")
        self.assertEqual((r["result"], r["reason"]), ("refused", "producer_binding"))


    def test_unstored_primitive_producer_refused(self):
        # add(mul(a, b), c) with mul unstored, computed as (a - b) + c: same leaves, other op.
        r = run("unstored_neg_sub", "unstored_prim")
        self.assertEqual((r["result"], r["reason"]), ("refused", "unstored_primitive_producer"))

    def test_multiplicand_matches(self):
        self.assertEqual(run("mult_pos", "mult")["result"], "matched")

    def test_multiplicand_bypass_refused(self):
        # a * b + a: a is a factor of an independent product on one path and bypasses it on another.
        r = run("mult_neg_bypass", "mult")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role"))
        self.assertEqual(r["detail"]["role"], "multiplicand")

    def test_chain_without_transcendental_matches(self):
        self.assertEqual(run("chain_pos", "chain")["result"], "matched")

    def test_path_through_another_producers_transcendental_refused(self):
        # C(P(a), Q(P(a))), Q = exp: p reaches C's output through Q, behind q's cut. A body
        # argument's paths end only at its own bound values.
        r = run("chain_neg_exp", "chain")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role"))
        self.assertEqual(r["detail"]["role"], "avoids_transcendental")

    def test_unstored_producer_converted_at_one_use_refused(self):
        # t at one use, f32(f16(t)) at the other: one node, two values.
        r = run("unstored_neg_conv", "unstored")
        self.assertEqual((r["result"], r["reason"]), ("refused", "producer_binding"))

    def test_multiplicand_on_a_body_argument_refused(self):
        r = run("body_mult", "body_mult")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role_boundary"))

    def test_path_through_a_transcendental_above_another_cut_refused(self):
        # Round-4 review reproducer: q = 2p stored, y = p * ex2(q). p reaches y through the ex2 above
        # q's cut; role traversal runs over the complete dataflow, so no cut ends that path.
        r = run("chain_neg_ex2", "chain", "handwritten")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role"))
        self.assertEqual(r["detail"]["role"], "avoids_transcendental")

    def test_stored_operand_of_the_same_component_matches(self):
        self.assertEqual(run("sel_pos_component", "sel", "handwritten")["result"], "matched")

    def test_stored_operand_of_another_component_refused(self):
        # p = f32(low half of a's bf16 word), y = f32(high half) + b: the operand has the stored
        # value's leaves and conversion chain, but is another value.
        r = run("sel_neg_component", "sel", "handwritten")
        self.assertEqual((r["result"], r["reason"]), ("refused", "primitive_operands"))

if __name__ == "__main__":
    unittest.main()
