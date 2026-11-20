"""Property-based differential test of the structural mode's role predicates.

Generates seeded random small terms and PTX kernels (hand-emitted PTX, no compiler: stored and
unstored producers, cuts read back through memory or from registers, f16 conversion round trips,
component selections of packed pairs, ex2/lg2, products, fmas and adds), runs the native checker
with --structural-reference and asserts: whenever the checker matches, the reference
implementation (plain reachability over the value DAG, cpp/src/value_analysis/structural.cpp
class Reference) decides every role check of the term and finds none failing.

Skipped unless SENTRYWARD_PTX_NATIVE names a built sentryward-ptx. SENTRYWARD_DIFF_CASES (default
400) and SENTRYWARD_DIFF_SEED (default 0xD263) set the size and seed; SENTRYWARD_DIFF_KEEP names a
directory that keeps every case (PTX, input, term, report).
"""
import json
import os
import random
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

NATIVE = os.environ.get("SENTRYWARD_PTX_NATIVE")
CASES = int(os.environ.get("SENTRYWARD_DIFF_CASES", "400"))
SEED = int(os.environ.get("SENTRYWARD_DIFF_SEED", str(0xD263)), 0)
THREADS = 32
ROLE_SETS = [[], [], ["through_transcendental"], ["avoids_transcendental"], ["addend"], ["multiplicand"],
             ["avoids_transcendental", "addend"], ["through_transcendental", "multiplicand"]]
IDENT = [[{"out": 0}, {"out": 1}]]


class Kernel:
    """Straight-line PTX: thread i reads and writes element i of each f32 object."""

    def __init__(self, name, objects):
        self.name, self.objects = name, objects
        self.body, self.f, self.h, self.d = [], 0, 0, 20
        self.base = {}
        # Per register, the generator's own dataflow model: the source tags some path reaches, and
        # those reached through ex2/lg2 above them. Used only to bias roles toward ones that hold.
        self.info = {}
        for i, o in enumerate(objects):
            self.body.append(f"ld.param.u64 %rd{i + 1}, [{name}_param_{i}];")
            self.body.append(f"cvta.to.global.u64 %rd{10 + i}, %rd{i + 1};")
            self.base[o] = f"%rd{10 + i}"
        self.body += ["mov.u32 %r1, %tid.x;", "mul.wide.s32 %rd9, %r1, 4;"]

    def freg(self):
        self.f += 1
        return f"%f{self.f}"

    def addr(self, o):
        self.d += 1
        self.body.append(f"add.s64 %rd{self.d}, {self.base[o]}, %rd9;")
        return f"%rd{self.d}"

    def load(self, o):
        a, r = self.addr(o), self.freg()
        self.body.append(f"ld.global.f32 {r}, [{a}];")
        return r

    def store(self, o, r):
        self.body.append(f"st.global.f32 [{self.addr(o)}], {r};")

    def op(self, text, n=1):
        rs = [self.freg() for _ in range(n)]
        self.body.append(text.format(*rs))
        return rs[0] if n == 1 else rs

    def text(self):
        params = ",\n".join(f"\t.param .u64 {self.name}_param_{i}" for i in range(len(self.objects)))
        regs = [f".reg .f32 %f<{self.f + 1}>;", f".reg .b16 %h<{self.h + 1}>;", ".reg .b32 %r<2>;",
                f".reg .b64 %rd<{self.d + 1}>;"]
        body = "\n".join("\t" + x for x in regs + [""] + self.body + ["ret;"])
        return (".version 8.8\n.target sm_90a\n.address_size 64\n\n"
                f".visible .entry {self.name}(\n{params}\n)\n{{\n{body}\n}}\n")


def expression(rng, k, leaves, depth):
    """A random f32 expression over the registers `leaves`, using each at least once."""
    def mk(text, xs, trans=False):
        r = k.op(text)
        alls = set().union(*(k.info[x][0] for x in xs)) if xs else set()
        thr = set().union(*(k.info[x][1] for x in xs)) if xs else set()
        k.info[r] = (alls, thr | (alls if trans else set()))
        return r

    def gen(d):
        if d == 0 or rng.random() < 0.25:
            return rng.choice(leaves)
        c = rng.random()
        if c < 0.22:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"add.f32 {{0}}, {x}, {y};", [x, y])
        if c < 0.42:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"mul.f32 {{0}}, {x}, {y};", [x, y])
        if c < 0.52:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"fma.rn.f32 {{0}}, {x}, {y}, 0f3F800000;", [x, y])
        if c < 0.62:
            x = gen(d - 1)
            return mk(f"add.f32 {{0}}, {x}, 0f40400000;", [x])
        if c < 0.72:
            x = gen(d - 1)
            return mk(f"ex2.approx.f32 {{0}}, {x};", [x], True)
        if c < 0.77:
            x = gen(d - 1)
            return mk(f"lg2.approx.f32 {{0}}, {x};", [x], True)
        if c < 0.88:  # f16 round trip: a conversion chain
            k.h += 1
            x = gen(d - 1)
            k.body.append(f"cvt.rn.f16.f32 %h{k.h}, {x};")
            return mk(f"cvt.f32.f16 {{0}}, %h{k.h};", [x])
        # a component of a packed pair (a selection)
        x, y = gen(d - 1), gen(d - 1)
        k.d += 1
        k.body.append(f"mov.b64 %rd{k.d}, {{{x}, {y}}};")
        lo, hi = k.op(f"mov.b64 {{{{{{0}}, {{1}}}}}}, %rd{k.d};", 2)
        k.info[lo], k.info[hi] = k.info[x], k.info[y]
        return rng.choice([lo, hi])
    e = gen(depth)
    for leaf in leaves:  # every operand appears
        if rng.random() < 0.85:
            e = mk(f"{rng.choice(['add', 'mul'])}.f32 {{0}}, {e}, {leaf};", [e, leaf])
    return e


def roles(rng, k, reg, tag):
    """A role set for the argument tagged `tag` of the atom whose value is `reg`: mostly the
    transcendental role the generator's model says holds, else a random set."""
    if rng.random() < 0.6:
        alls, thr = k.info[reg]
        if tag in alls:
            return ["through_transcendental" if tag in thr else "avoids_transcendental"]
    return rng.choice(ROLE_SETS)


def obj(name):
    return {"name": name, "element": "f32", "element_bytes": 4, "base_bytes": 0,
            "row_stride_bytes": 4 * THREADS, "rows": 1, "cols": THREADS, "row_indexed": False}


def opaque(op, args, roles, out):
    return {"op": op, "atom": {"kind": "opaque", "roles": roles}, "format": {"compute": None, "store": "f32"},
            "args": args, "out": out}


def case(rng, name):
    """A random (term, kernel) pair. Shapes: stored producer chains under an opaque consumer, and
    an unstored opaque producer under primitive adds."""
    if rng.random() < 0.7:
        two = rng.random() < 0.5
        objects = ["a", "b", "p"] + (["q"] if two else []) + ["y"]
        k = Kernel(name, objects)
        a, b = k.load("a"), k.load("b")
        k.info[a], k.info[b] = ({"a"}, set()), ({"b"}, set())
        p = expression(rng, k, [a], rng.randint(0, 3))
        k.store("p", p)

        def reread(o, r):  # a cut read back through memory or used from its register
            x = k.load(o) if rng.random() < 0.5 else r
            k.info[x] = (k.info[r][0] | {o}, k.info[r][1])
            return x
        nodes = [opaque("P", [{"input": 0, "reads": IDENT}], [roles(rng, k, p, "a")], "p")]
        regs = {"p": p}
        if two:
            q = expression(rng, k, [reread("p", p)] + ([a] if rng.random() < 0.2 else []), rng.randint(0, 3))
            k.store("q", q)
            regs["q"] = q
            nodes.append(opaque("Q", [{"body": 0, "reads": IDENT}], [roles(rng, k, q, "p")], "q"))
        pool = [("body", 0, "p")] + ([("body", 1, "q")] if two else []) + [("input", 0, "a"), ("input", 1, "b")]
        args = rng.sample(pool, rng.randint(1, min(3, len(pool))))
        leaves = [reread(n, regs[n]) if kind == "body" else (a if n == "a" else b) for kind, _, n in args]
        y = expression(rng, k, leaves, rng.randint(0, 3))
        k.store("y", y)
        nodes.append(opaque("C", [{kind: ref, "reads": IDENT} for kind, ref, _ in args],
                            [roles(rng, k, y, n) for _, _, n in args], "y"))
        term = {"objects": [obj(o) for o in objects], "inputs": ["a", "b"], "nodes": nodes}
    else:
        uses = rng.randint(1, 2)
        objects = ["a", "b", "c"] + [f"y{i}" for i in range(uses)]
        k = Kernel(name, objects)
        a, b, c = k.load("a"), k.load("b"), k.load("c")
        k.info[a] = ({"a"}, set())
        t = expression(rng, k, [a], rng.randint(1, 3))
        nodes = [opaque("P", [{"input": 0, "reads": IDENT}], [roles(rng, k, t, "a")], None)]
        for i in range(uses):
            u = t if rng.random() < 0.8 else expression(rng, k, [a], rng.randint(1, 2))
            k.store(f"y{i}", k.op(f"add.f32 {{0}}, {u}, {[b, c][i]};"))
            nodes.append({"op": "add", "atom": {"kind": "primitive", "operation": "add", "commutative": True},
                          "format": {"compute": None, "store": "f32"},
                          "args": [{"body": 0, "reads": IDENT}, {"input": 1 + i, "reads": IDENT}], "out": f"y{i}"})
        term = {"objects": [obj(o) for o in objects], "inputs": ["a", "b", "c"], "nodes": nodes}
    term.update({"schema": "sentryward.structural-term/v2", "row": f"diff/{name}", "template": f"diff/{name}"})
    inputs = {"schema": "spverify.kernel-input/v1", "entry": name,
              "launch": {"grid_dim": [1, 1, 1], "block_dim": [THREADS, 1, 1], "dynamic_shared_bytes": 0},
              "declarations": [{"parameter": i, "type": "pointer", "space": "global", "object": o}
                               for i, o in enumerate(objects)], "arguments": []}
    return term, inputs, k.text()


def run_case(native, d, term, inputs, ptx):
    d.mkdir(parents=True, exist_ok=True)
    (d / "k.ptx").write_text(ptx)
    (d / "in.json").write_text(json.dumps(inputs))
    (d / "term.json").write_text(json.dumps(term))
    subprocess.run([native, "--ptx", str(d / "k.ptx"), "--config", str(d / "in.json"), "--analyze-scalar-fold",
                    "--structural-term", str(d / "term.json"), "--structural-report", str(d / "report.json"),
                    "--structural-reference"], capture_output=True, text=True, check=False, timeout=120)
    return json.loads((d / "report.json").read_text())


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class StructuralDifferentialTest(unittest.TestCase):
    def test_checker_never_accepts_where_the_reference_refuses(self):
        rng = random.Random(SEED)
        keep = os.environ.get("SENTRYWARD_DIFF_KEEP")
        outcomes, role_checks, violations = Counter(), 0, []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(keep or tmp)
            for i in range(CASES):
                name = f"diff_{i}"
                term, inputs, ptx = case(rng, name)
                r = run_case(NATIVE, root / name, term, inputs, ptx)
                outcomes[r["result"] if r["result"] == "matched" else f'{r["result"]}:{r.get("reason")}'] += 1
                if r["result"] != "matched":
                    continue
                ref = r["reference"]
                role_checks += ref["checks"]
                if ref["fails"] or ref["undecided"]:
                    violations.append({"case": i, "fails": ref["fails"][:3], "undecided": ref["undecided"][:3]})
        print(f"\nseed {SEED:#x}, {CASES} cases: {dict(outcomes)}; role checks in matched cases: {role_checks}")
        self.assertEqual(violations, [], f"checker matched where the reference refuses (seed {SEED:#x})")
        # Not vacuous: both outcomes occur, and matched cases carry role checks.
        self.assertGreater(outcomes["matched"], CASES // 10)
        self.assertGreater(outcomes["refused:role"], CASES // 20)
        self.assertGreater(role_checks, 0)


if __name__ == "__main__":
    unittest.main()
