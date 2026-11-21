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

    def test_field_premise_bounds_one_record_field(self):
        # Word 3 of each 16-byte record (after a 32-byte header) in [0, 4): the gathered row is
        # inside hs; the header words stay unconstrained.
        r = run("gather", "gather_field_premise")
        fp = self.footprint(r)
        self.assertEqual((fp["writes"], fp["reads"]), ("contained", "contained"))

    def test_field_premise_widened_refused(self):
        fp = self.footprint(run("gather", "gather_field_premise_widened"))
        self.assertEqual(fp["writes"], "violated")
        self.assertIn("meta:s32(44)=4", fp["violations"][0]["witness"])

    def test_field_premise_on_another_word_is_no_premise(self):
        # Bounding word 2 instead of word 3 leaves the sample unbounded: the branch on it is not
        # decided, never a pass.
        r = run("gather", "gather_field_premise_wrong_word")
        self.assertEqual(r["result"], "unsupported")
        self.assertNotEqual(r["checks"].get("footprint"), "passed")

    def test_field_reads_one_element_per_cta(self):
        # k_gather's block r reads record r's sample word and no other element of the field.
        r = run("gather", "gather_field_premise")
        self.assertEqual(r["frontend_assumptions"]["field_reads"],
                         [{"object": "meta", "field": {"offset": 44, "stride": 16, "count": 4},
                           "max_elements_per_cta": 1}])

    def test_field_reads_two_elements_reported(self):
        # Planted mutant: block r also loads record r+1's sample word.
        r = run("gather_two_reads", "gather_field_premise")
        (reads,) = r["frontend_assumptions"]["field_reads"]
        self.assertEqual(reads["max_elements_per_cta"], 2)

    # Branch-path conditions (k_feed: meta[8 + 4 t] = w - 1 < slots ? next_tok[w - 1] : NONE, with
    # w = feed[t] loaded data). On the path that takes none of the three exits, the read of
    # next_tok[w - 1] is under the guard w - 1 < slots (slots = 8, next_tok 32 bytes): contained.
    def test_path_condition_bounds_guarded_read(self):
        r = run("feed", "feed_guarded_read")
        fp = self.footprint(r)
        self.assertEqual(r["result"], "race_free")
        self.assertEqual((fp["writes"], fp["reads"]), ("contained", "contained"))

    # The same path with the guard weakened to w - 1 <= slots: next_tok[slots] is read.
    def test_path_condition_off_by_one_guard_reads_outside(self):
        r = run("feed_guard_off_by_one", "feed_guarded_read")
        fp = self.footprint(r)
        self.assertEqual(fp["reads"], "not_contained")
        self.assertEqual(fp["outside_reads"][0]["object"], "next_tok")

    # k_sample folds `__syncthreads_or(bad)` (bar.red.or.pred) into its pick: a CTA barrier whose
    # result (the same in every thread) depends on loaded data. It is checked, not refused as an
    # unsupported collective.
    def test_bar_red_or_is_a_barrier_with_an_unknown_uniform_result(self):
        r = run("feed", "sample_reduce_or")
        # The leaf runs past the barrier to the kernel's next data-dependent branch (the choice
        # tree expands it), instead of stopping at the collective.
        codes = [(str(d.get("code")), (d.get("details") or {}).get("location", {}).get("line")) for d in r.get("diagnostics", [])]
        self.assertNotIn("shared_control_collective_variant", [c for c, _ in codes])
        self.assertIn(("control_predicate_not_static", 1492), codes)

    # The guard's predicate as the complement destination (`setp.ge.u32 _|%p4, w - 1, slots`):
    # %p4 is w - 1 < slots, so the not-taken path reads next_tok[w - 1] with w - 1 >= slots.
    # The comparison must not be read with the first destination's polarity.
    def test_path_condition_complement_destination_reads_outside(self):
        r = run("feed_guard_complement", "feed_guarded_read")
        fp = self.footprint(r)
        self.assertEqual(fp["reads"], "not_contained")
        self.assertEqual(fp["outside_reads"][0]["object"], "next_tok")


    # Stage A, relational path conditions (docs/design/loop-summaries.md; src/path.cu). Each case
    # is one thread; the access sits on the not-taken side of the kernel's one data branch.
    def writes(self, inp):
        return self.footprint(run("path", inp))["writes"]

    def test_path_eq_bounds_exactly(self):
        self.assertEqual(self.writes("path_eq"), "contained")

    def test_path_eq_past_the_end_refused(self):
        self.assertEqual(self.writes("path_eq_past"), "violated")

    def test_path_two_unknowns_relational(self):
        self.assertEqual(self.writes("path_two"), "contained")

    def test_path_two_unknowns_off_by_one_refused(self):
        self.assertEqual(self.writes("path_two_past"), "violated")

    def test_path_or_pred_not_taken_bounds_both(self):
        self.assertEqual(self.writes("path_or"), "contained")

    def test_path_or_pred_with_known_operand(self):
        self.assertEqual(self.writes("path_or_known"), "contained")

    def test_path_or_pred_with_known_operand_off_by_one_refused(self):
        self.assertEqual(self.writes("path_or_known_past"), "violated")

    def test_path_wrapped_equality_reads_before_the_object(self):
        self.assertEqual(self.writes("path_eq16_wrapped"), "violated")

    def test_piece_rows_straddling_store_not_contained(self):
        self.assertNotEqual(self.writes("path_index_rows"), "contained")

    def test_path_facts_are_lane_scoped(self):
        # Lane 0's x < 8 on the shared element must not bound lane 1's unguarded out[x].
        self.assertEqual(self.writes("path_cross_lane"), "violated")

    # Stage C: a disjunctive meaning stops the run (fact_disjunction, with its number of cases);
    # each case is then a run of its own (fact_cases). Cases contradicting the lane's facts are
    # vacuous for the lane.
    def case_writes(self, inp):
        base = json.loads((HERE / "inputs" / f"{inp}.json").read_text())
        first = run("path", inp)
        stop = [d for d in first.get("diagnostics", []) if d.get("code") == "fact_disjunction"]
        self.assertEqual(len(stop), 1, first.get("diagnostics"))
        line = stop[0]["details"]["location"]["line"]
        n = stop[0]["details"]["required_parameters"]["cases"]
        self.assertGreater(n, 1)
        out = []
        for k in range(n):
            tmp = HERE / "inputs" / f".case_{os.getpid()}.json"
            tmp.write_text(json.dumps(dict(base, fact_cases=[{"line": line, "case": k, "cases": n}])))
            try:
                out.append(self.footprint(run("path", tmp.stem))["writes"])
            finally:
                tmp.unlink()
        return out

    def test_path_disjunction_cases_contained(self):
        self.assertEqual(set(self.case_writes("path_wrap_cases")), {"contained"})

    def test_path_disjunction_cases_off_by_one_refused(self):
        self.assertIn("violated", self.case_writes("path_wrap_cases_small"))

    def test_path_and_pred_not_taken_bounds_neither(self):
        # Not (s >= 8 && last) is (s < 8) || !last: a case split (stage C); the !last case leaves s
        # unbounded, so some case is violated.
        self.assertIn("violated", self.case_writes("path_and"))

    def test_path_low_bit_mask_inside(self):
        self.assertEqual(self.writes("path_mask"), "contained")

    def test_path_other_mask_past_the_end_refused(self):
        self.assertEqual(self.writes("path_mask_odd"), "violated")

    # Stage B, loop summaries (docs/design/loop-summaries.md; src/loop.cu, #pragma unroll 1 except
    # k_copy_unrolled). A run that stops at a data branch outside a loop is split into a choice tree
    # (both outcomes, as coral's run.py does); every leaf must be decided.
    def tree(self, inp, budget=24):
        base = json.loads((HERE / "inputs" / f"{inp}.json").read_text())
        leaves, pending = [], [[]]
        while pending:
            path = pending.pop()
            self.assertLess(len(leaves) + len(pending), budget, "choice tree budget")
            body = dict(base, branch_decisions=path) if path else base
            tmp = HERE / "inputs" / f".tree_{os.getpid()}.json"
            tmp.write_text(json.dumps(body))
            try:
                r = run("loop", tmp.stem)
            finally:
                tmp.unlink()
            diag = [(d.get("code"), (d.get("details") or {}).get("location", {}).get("line")) for d in r.get("diagnostics", [])]
            stop = next((line for code, line in diag if code == "control_predicate_not_static"), None)
            if stop is not None and not any(c["line"] == stop for c in path) and "footprint" not in r.get("evidence", {}):
                pending += [path + [{"line": stop, "taken": True}], path + [{"line": stop, "taken": False}]]
                continue
            leaves.append((path, r))
        return leaves

    def leaf_status(self, r):
        fp = r.get("evidence", {}).get("footprint")
        if fp is None:
            return r["result"]
        if any("unsatisfiable" in str(d.get("message", "")) for d in r.get("diagnostics", [])):
            return "infeasible"
        return (fp["writes"], fp["reads"])

    def test_loop_counted_contained(self):
        for path, r in self.tree("loop_copy_n"):
            self.assertIn(self.leaf_status(r), [("contained", "contained"), "infeasible"], path)

    def test_loop_off_by_one_refused(self):
        statuses = [self.leaf_status(r) for _, r in self.tree("loop_copy_le")]
        self.assertTrue(any(s != "infeasible" and "violated" in s for s in statuses), statuses)

    def test_loop_havocked_index_not_contained(self):
        statuses = [self.leaf_status(r) for _, r in self.tree("loop_chase")]
        self.assertTrue(any(isinstance(s, tuple) and s[1] != "contained" for s in statuses), statuses)

    # Codex review of stage B: an exported induction variable (i == n >= 1 at exit, n a tensor-property
    # element) guards a store past the extent; its exit range must come from the element's property.
    def test_loop_exported_iv_guards_store_outside(self):
        r = run("exit_iv", "loop_exit_iv")
        self.assertEqual(r["frontend_assumptions"]["loop_summaries"], 32)
        self.assertEqual(self.footprint(r)["writes"], "violated")

    def test_loop_rows_from_table_contained(self):
        for path, r in self.tree("loop_rows"):
            self.assertIn(self.leaf_status(r), [("contained", "contained"), "infeasible"], path)

if __name__ == "__main__":
    unittest.main()
