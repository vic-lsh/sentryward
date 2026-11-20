"""CTA-row shards (docs/design/cta-row-shards.md) end to end on the native executable.

Input `cta_rows: {y: [lo, hi)}` executes only the CTAs of those grid rows under the launch's true
grid and words, after the shard allowlist scan admits the kernel (entry and direct callees). The
footprint report then lists, per object, the bytes the shard's accesses touched (constant offsets
exact, anything else counted as symbolic), and `frontend_assumptions.cta_rows` the launch identity
(the input as given without cta_rows, the PTX digest) that the caller authenticates.

The PTX in tests/cta_rows_cases is hand-written: a row copy (admitted), cross-row and symbolic
access shapes whose pieces the cross-shard check (Coral's check_verdicts.py) decides, and one
kernel per refused form (placement or time registers, a global atomic, a bulk tensor copy, a
barrier, a non-coherent or generic access, a call to a refused function). Skipped unless
SENTRYWARD_PTX_NATIVE names a built sentryward-ptx.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent / "cta_rows_cases"
ROOT = Path(__file__).resolve().parents[1]
NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")

ROW = 128  # bytes per grid row (32 threads x 4 bytes)
ROWS = 4


def launch(entry, rows=(1, 3), symbolic=False):
    inp = {
        "schema": "spverify.kernel-input/v1",
        "entry": entry,
        "launch": {"grid_dim": [1, ROWS, 1], "block_dim": [32, 1, 1]},
        "declarations": [{"parameter": 0, "type": "pointer", "space": "global", "object": "out"},
                         {"parameter": 1, "type": "pointer", "space": "global", "object": "src"}],
        "footprints": [{"object": "out", "direction": "readwrite", "extent_bytes": (ROWS + 1) * ROW},
                       {"object": "src", "direction": "read", "extent_bytes": ROWS * ROW}],
        "footprint_reads": "noninterference",
    }
    if symbolic:
        field = {"offset": 0, "stride": 8, "count": ROWS}
        inp["tensor_properties"] = [
            {"object": "src", "element_type": "s64", "property": "lower_bound", "value": 0, "field": field},
            {"object": "src", "element_type": "s64", "property": "upper_bound", "value": ROWS - 1, "field": field}]
    if rows is not None:
        inp["cta_rows"] = {"y": list(rows)}
    return inp


def run(entry, inp):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(inp, f)
    try:
        out = subprocess.run([sys.executable, "-m", "sentryward", str(HERE / "ptx" / f"{entry}.ptx"),
                              "--input", f.name, "--timeout", "300"],
                             capture_output=True, text=True, env=env, check=False)
    finally:
        os.unlink(f.name)
    return json.loads(out.stdout)


def diagnostic(r):
    return (r.get("diagnostics") or [{}])[0].get("code")


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class CtaRowsNativeTest(unittest.TestCase):
    def pieces(self, r):
        return r["evidence"]["footprint"]["pieces"]

    def test_shard_executes_only_its_rows(self):
        r = run("rows", launch("rows", (1, 3)))
        self.assertEqual(r["result"], "race_free")
        p = self.pieces(r)
        self.assertEqual(p["unsized_accesses"], 0)
        # Rows 1 and 2 only, at their launch addresses (no renumbering).
        self.assertEqual(p["objects"]["out"]["writes"], [{"base": ROW, "rows": 1, "stride": 0, "width": 2 * ROW}])
        self.assertEqual(p["objects"]["src"]["reads"], [{"base": ROW, "rows": 1, "stride": 0, "width": 2 * ROW}])
        self.assertEqual(p["objects"]["out"]["symbolic_writes"], 0)

    def test_identity_is_the_input_without_cta_rows(self):
        inp = launch("rows", (0, 2))
        r = run("rows", inp)
        shard = r["frontend_assumptions"]["cta_rows"]
        self.assertEqual(shard["y"], [0, 2])
        self.assertTrue(shard["allowlist"]["admitted"])
        want = dict(inp)
        del want["cta_rows"]
        self.assertEqual(shard["identity"]["input"], want)
        ptx = (HERE / "ptx" / "rows.ptx").read_bytes()
        self.assertEqual(shard["identity"]["ptx_sha256"], hashlib.sha256(ptx).hexdigest())
        # The grid is the launch's: grid.y stays 4 in the identity.
        self.assertEqual(shard["identity"]["input"]["launch"]["grid_dim"], [1, ROWS, 1])

    def test_cross_row_read_is_exported(self):
        # Row y writes out[y] and reads out[y + 1]: one row alone is race free; both pieces are
        # reported, so the cross-shard check can see row 1's read meet row 2's write.
        p = self.pieces(run("rows_next", launch("rows_next", (1, 2))))
        self.assertEqual(p["objects"]["out"]["writes"], [{"base": ROW, "rows": 1, "stride": 0, "width": ROW}])
        self.assertEqual(p["objects"]["out"]["reads"], [{"base": 2 * ROW, "rows": 1, "stride": 0, "width": ROW}])

    def test_symbolic_write_is_reported(self):
        r = run("symbolic_write", launch("symbolic_write", (0, 2), symbolic=True))
        p = self.pieces(r)
        self.assertGreater(p["objects"]["out"]["symbolic_writes"], 0)
        self.assertEqual(p["objects"]["out"]["writes"], [])

    def test_refused_forms(self):
        refused = {"smid": "%smid", "clock": "%clock", "atom": "atom.global.gpu.add.u32",
                   "bulk": "cp.async.bulk.tensor", "barrier": "bar.sync", "nc": "ld.global.nc.u32",
                   "generic": "st.u32", "call_smid": "peek: not on the shard allowlist: mov.u32 %r1, %smid"}
        for entry, what in refused.items():
            with self.subTest(entry=entry):
                r = run(entry, launch(entry))
                self.assertEqual(r["result"], "unsupported")
                self.assertEqual(diagnostic(r), "cta_rows_not_admitted", r.get("diagnostics"))
                self.assertIn(what, r["diagnostics"][0]["message"])

    def test_direct_call_to_listed_function_passes_the_scan(self):
        # The scan follows the call into `noop` and admits it; the frontend then stops at the call
        # itself (a call with a body has no summary), which is its existing limit, not the scan's.
        r = run("call_ok", launch("call_ok"))
        self.assertNotEqual(diagnostic(r), "cta_rows_not_admitted")
        self.assertEqual(diagnostic(r), "ptx_execution_not_supported")

    def test_without_cta_rows_nothing_changes(self):
        r = run("smid", launch("smid", rows=None))
        self.assertNotIn("cta_rows", r.get("frontend_assumptions", {}))
        self.assertNotIn("pieces", r.get("evidence", {}).get("footprint", {}))

    def test_rows_outside_the_grid_refused(self):
        for rows, code in (((2, 5), "invalid_cta_rows"), ((2, 2), "invalid_cta_rows")):
            with self.subTest(rows=rows):
                r = run("rows", launch("rows", rows))
                self.assertIn(code, json.dumps(r))


# Coral's H100 k_stage_in (stage.cu, nvcc 12.9, sm_90a): the opcode forms the design note's audit
# lists. The test re-derives them from the PTX text, so a recompiled kernel with another form fails
# here before any shard is trusted.
AUDITED = {"add.s64", "sub.s64", "mul.lo.s64", "mul.wide.u32", "div.u32", "div.u64", "and.b64", "or.b64",
           "shl.b64", "cvt.u32.u64", "cvt.u64.u32", "cvta.to.global.u64", "mov.u32", "setp.eq.s64",
           "setp.ne.s64", "setp.lt.u64", "setp.gt.u64", "bra", "bra.uni", "ret", "ld.param.u64",
           "ld.global.u64", "ld.global.v4.u32", "st.global.v4.u32"}
AUDITED_SPECIALS = {"%tid.x", "%ntid.x", "%ctaid.x", "%ctaid.y", "%nctaid.x"}


def entry_forms(ptx, entry):
    import re
    body = ptx.split(f".entry {entry}(", 1)[1].split("\n}", 1)[0]
    forms, specials = set(), set()
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith((".", "$", "{", "}", ")", "//")):
            continue
        line = re.sub(r"^@!?%p\d+\s+", "", line)
        forms.add(line.split()[0].rstrip(";"))
        specials |= set(re.findall(r"%(?:tid|ntid|ctaid|nctaid|laneid|smid|clock\w*|globaltimer\w*|warpid|gridid|\w*cluster\w*)\.?[xyz]?", line))
    return forms, specials


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class StageInAuditTest(unittest.TestCase):
    def test_audit_forms(self):
        forms, specials = entry_forms((HERE / "ptx" / "stage_sm90a.ptx").read_text(), "k_stage_in")
        self.assertEqual(forms, AUDITED)
        self.assertEqual(specials, AUDITED_SPECIALS)

    def test_one_row_shard(self):
        # One of 48 rows of the H100 launch (512 x 48 CTAs of 256 threads, 2 MiB per row).
        row, stride = 2097152, 64880640
        inp = {"schema": "spverify.kernel-input/v1", "entry": "k_stage_in",
               "launch": {"grid_dim": [512, 48, 1], "block_dim": [256, 1, 1], "dynamic_shared_bytes": 0},
               "declarations": [{"parameter": 0, "type": "pointer", "space": "global", "object": "tab"},
                                {"parameter": 3, "type": "pointer", "space": "global", "object": "state"},
                                {"parameter": 5, "type": "pointer", "space": "global", "object": "staging"}],
               "arguments": [{"parameter": 1, "type": "u64", "value": 1}, {"parameter": 2, "type": "u64", "value": 0},
                             {"parameter": 4, "type": "u64", "value": 253440}],
               "tensor_properties": [
                   {"object": "tab", "element_type": "s64", "property": p, "value": v,
                    "field": {"offset": 0, "stride": 8, "count": 48}}
                   for p, v in (("lower_bound", 0), ("upper_bound", 48))],
               "footprints": [
                   {"object": "staging", "direction": "write", "extent_bytes": 48 * row,
                    "pieces": [{"base": 0, "rows": 48, "stride": row, "period": row, "count": 1, "width": row,
                                "direction": "write"}]},
                   {"object": "state", "direction": "read", "extent_bytes": 97 * stride,
                    "pieces": [{"base": 0, "rows": 97, "stride": stride, "period": row, "count": 1, "width": row,
                                "direction": "read"}]},
                   {"object": "tab", "direction": "read", "extent_bytes": 384,
                    "pieces": [{"base": 0, "rows": 48, "stride": 8, "period": 8, "count": 1, "width": 8,
                                "direction": "read"}]}],
               "footprint_reads": "noninterference", "cta_rows": {"y": [5, 6]}}
        r = run("stage_sm90a", inp)
        self.assertEqual(r["result"], "race_free")
        self.assertEqual(r["frontend_assumptions"]["cta_rows"]["allowlist"],
                         {"admitted": True, "scanned": ["k_stage_in"]})
        o = r["evidence"]["footprint"]["pieces"]["objects"]
        self.assertEqual(o["staging"]["writes"], [{"base": 5 * row, "rows": 1, "stride": 0, "width": row}])
        self.assertEqual(o["tab"]["reads"], [{"base": 40, "rows": 1, "stride": 0, "width": 8}])
        self.assertEqual((o["state"]["reads"], o["state"]["symbolic_reads"] > 0), ([], True))


if __name__ == "__main__":
    unittest.main()
