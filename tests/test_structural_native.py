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
# The IR-level controls (structural_cases/ir/index_controls.cpp, target
# sentryward-structural-ir-controls, built on request): SENTRYWARD_STRUCTURAL_IR, else next to NATIVE.
IR = os.environ.get("SENTRYWARD_STRUCTURAL_IR") or (
    str(Path(NATIVE).parent / "sentryward-structural-ir-controls") if NATIVE else None)


def run(entry, term, ptx="roles", reference=False):
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run([NATIVE, "--ptx", str(HERE / "ptx" / f"{ptx}.ptx"), "--config",
                        str(HERE / "inputs" / f"{entry}.json"), "--analyze-scalar-fold",
                        "--structural-term", str(HERE / "terms" / f"{term}.json"),
                        "--structural-report", str(report)] + (["--structural-reference"] if reference else []),
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

    def test_stored_operand_of_a_word_half_refused(self):
        # p = f32(low half of a's bf16 word): the half uses part of the word's bits, so neither
        # component is a full dependency on the word's leaves (was matched before partial_operand;
        # identity by component is covered by the IR-level operand control).
        for entry in ("sel_pos_component", "sel_neg_component"):
            r = run(entry, "sel", "handwritten")
            self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"), entry)

    def test_partial_pack_refused(self):
        # Round-5 reproducer: P stores (a & 0xffff) | (b & 0xffff0000); C reads P's low half, so y
        # ignores b. P itself uses a and b only in part (partial_operand, checked first); the
        # partial read alone is control partial_stored_half (partial.ptx).
        r = run("pack", "pack", "pack")
        self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"))

    # Partial-operand controls (ptx/partial.ptx): p = a + b stored, reloaded at full width, one
    # operation, its bits stored as y; term P(a, b) -> p, C(p) -> y. Every negative matched at
    # 7b4673b (before partial_operand); the reference there was undecided (it did not follow a
    # same-width reload of a stored value) and now refutes each by its own rule.
    PARTIAL = ["and_mask", "or_mask", "shl", "shr_u", "shr_s_past", "shift_mixed", "bfe", "bfi", "prmt", "lop3",
               "shf", "cvt_narrow", "unpack", "copysign", "selp_narrow", "abs", "mul_even", "data_mask",
               "round_low", "round_no_sign", "round_short",
               # Round-6 review: an integer product whose other factor's trailing zeros are not
               # decided zero (p * (p << 1) = 2p^2 ignores p's bit 31; also mad.lo, a factor with
               # its low bit fixed zero, a quotient by data), and p's float bits >> 16 used as an
               # f32 (0x00003f80 for 1.0f: not a narrowing). round_bf16_shr (>> 16 then << 16, an
               # f32) is refused too: a shift is not a rounding, only a narrowing store is (narrow.ptx).
               "mul_var", "mul_var_even", "mad_var", "div_var", "shr_as_f32", "round_bf16_shr"]
    # round_bf16: and 0xffff0000 on p's own float bits clears the low mantissa in place: p rounded
    # toward zero, in f32. mul_odd (p * 3), mul_var_odd (p * (p | 1)): the other factor is odd.
    FULL = ["xor_full", "prmt_full", "lop3_full", "round_bf16", "mul_odd", "mul_var_odd"]

    def test_partial_operands_refused_by_checker_and_reference(self):
        for name in self.PARTIAL:
            r = run(f"partial_{name}", "pack", "partial", reference=True)
            self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"), name)
            leaves = r["reference"]["leaves"]
            self.assertGreater(len(leaves["fails"]), 0, name)  # the reference's own rule refutes it
            self.assertEqual(leaves["undecided"], [], name)

    def test_full_operands_match(self):
        for name in self.FULL:
            r = run(f"partial_{name}", "pack", "partial", reference=True)
            self.assertEqual(r["result"], "matched", name)
            ref = r["reference"]
            self.assertEqual([ref[s]["fails"] + ref[s]["undecided"] for s in ("leaves", "cuts")], [[], []], name)

    # Narrowing controls (ptx/narrow.ptx): p = a + b stored as f32, one 16-bit value per thread
    # stored into y. Term pack_bf16 (y a bf16 field) or pack_f16 (an f16 field).
    # Positives: p's float bits >> 16 stored as u16 (k_combine_pos_store_trunc's PTX: mov.b32,
    # shr.u32 16, st.global.u16) and shr.s32 (the same 16 stored bits), cvt.rn and cvt.rz to bf16.
    NARROW_FULL = ["shr_store", "shr_signed", "cvt_rn", "cvt_rz"]
    # Negatives: >> 17 (not the bf16 encoding), >> 16 of p reloaded as u32 (an integer view, not
    # p's float bits); and shr_store into an f16 field (pack_f16).
    NARROW_PARTIAL = ["shr17", "shr_reload"]

    def test_narrowing_store_is_a_rounding(self):
        for name in self.NARROW_FULL:
            r = run(f"narrow_{name}", "pack_bf16", "narrow", reference=True)
            self.assertEqual(r["result"], "matched", name)
            ref = r["reference"]
            self.assertEqual([ref[s]["fails"] + ref[s]["undecided"] for s in ("leaves", "cuts")], [[], []], name)

    def test_other_top_half_uses_refused(self):
        cases = [(f"narrow_{n}", "pack_bf16") for n in self.NARROW_PARTIAL] + [("narrow_shr_store", "pack_f16")]
        for entry, term in cases:
            r = run(entry, term, "narrow", reference=True)
            self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"), (entry, term))
            leaves = r["reference"]["leaves"]
            self.assertGreater(len(leaves["fails"]), 0, entry)
            self.assertEqual(leaves["undecided"], [], entry)

    def test_reference_undecided_outside_exact_probes(self):
        # Round-6 ruling (c): y = p & ((p>>2 ^ p>>3) & (p ^ p>>1)). Each mask bit depends on four of
        # p's bits and is zero at every reference probe; the reference reports undecided (never
        # agreement), the checker refuses the data mask.
        r = run("partial_four_bit_mask", "pack", "partial", reference=True)
        self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"))
        leaves = r["reference"]["leaves"]
        self.assertEqual(leaves["fails"], [])
        self.assertIn("more than three free bits", leaves["undecided"][0]["why"])

    def test_partial_stored_read_refused(self):
        # p = a + b stored, its low half read back as u16 (memory, not registers).
        r = run("partial_stored_half", "pack", "partial")
        self.assertEqual((r["result"], r["reason"]), ("refused", "partial_stored_read"))

    def test_product_with_zero_refused(self):
        # Round-5 reproducer: y = a * b + 0 * ex2(a) is not a path from a through ex2.
        r = run("mult_pos", "silu_mul", "dead")
        self.assertEqual((r["result"], r["reason"]), ("refused", "dead_operand"))

    def test_and_with_zero_refused(self):
        r = run("dead_and", "silu_mul", "dead_paths")
        self.assertEqual((r["result"], r["reason"]), ("refused", "dead_operand"))

    def test_shift_past_width_refused(self):
        r = run("dead_shl", "silu_mul", "dead_paths")
        self.assertEqual((r["result"], r["reason"]), ("refused", "dead_operand"))

    def test_select_with_fixed_condition_not_credited(self):
        # selp(ex2(a), 0, tid < 0): the condition is fixed per thread, the select resolves to 0, so
        # no path from a passes ex2 and the through role fails.
        r = run("dead_sel", "silu_mul", "dead_paths")
        self.assertEqual((r["result"], r["reason"]), ("refused", "role"))
        self.assertEqual(r["detail"]["role"], "through_transcendental")

    def test_mask_on_the_only_transcendental_path_refused(self):
        # and with 0x7fffffff (|ex2(a)|): not a dead edge, but it uses all of ex2(a)'s bits except
        # the sign, so the only path from a through ex2 is partial and the through role does not
        # hold in full (matched before partial_operand).
        r = run("live_and", "silu_mul", "dead_paths")
        self.assertEqual((r["result"], r["reason"]), ("refused", "partial_operand"))


def run_ir(case, polarity):
    out = subprocess.run([IR, case, polarity], capture_output=True, text=True, check=True, timeout=60).stdout
    return json.loads(out)


@unittest.skipUnless(IR and Path(IR).exists(), "sentryward-structural-ir-controls is not built")
class StructuralIrIndexTest(unittest.TestCase):
    """Pairs that differ only in one value's index (structural_cases/ir/index_controls.cpp)."""

    def check(self, case, reason):
        self.assertEqual(run_ir(case, "positive")["result"], "matched")
        r = run_ir(case, "negative")
        self.assertEqual((r["result"], r["reason"]), ("refused", reason))

    def test_cut_identity_by_index(self):
        self.check("factorization", "factorization")

    def test_stored_operand_identity_by_index(self):
        self.check("operand", "primitive_operands")

    def test_producer_binding_identity_by_index(self):
        self.check("binding", "producer_binding")


if __name__ == "__main__":
    unittest.main()
