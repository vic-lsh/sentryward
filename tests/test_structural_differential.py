"""Property-based differential tests of the structural mode.

Two seeded generators, each with an oracle independent of the checker:

1. PTX cases (sentryward-ptx --structural-reference): hand-emitted PTX, no compiler. Shapes:
   stored producer chains under an opaque consumer (cuts read back through memory or used from
   registers), also over a grid of CTAs with row-indexed objects (class mode); an unstored opaque
   producer under primitive adds; a warp-shuffle reduction (shfl.sync.bfly butterfly) read as a
   whole row, sometimes with a step missing. Expressions: f16 round trips, packed-pair components,
   selects on data-dependent conditions, ex2/lg2, products, fmas, adds, and masked operands (a
   value's bits through and/shr/shl/prmt/bfe or an integer product with an even factor that keep
   part of them, a shift right by 16 used as an f32 (a fake narrowing), or xor/prmt, an in-place
   and 0xffff0000 or a product with an odd factor that keep all).
2. IR cases (sentryward-structural-ir-controls random): value IR built directly, with indexed
   selections (input elements are components of matrix loads), stored and unstored producers and
   mutations whose outcome the generator knows (an index, a recomputed cut, a second binding, a
   wrong role).

Oracles. The reference (structural.cpp class Reference, --structural-reference) restates the
leaf, cut, binding and role checks by plain reachability with its own term-element, load, output
and binding mapping. Each generator also keeps its own dataflow model (sources reached, and
through ex2/lg2), from which it knows whether a transcendental role holds and whether a mutation
must be refused, and which sources it reaches only through partial masks (a stored value or a
role that needs one in full must be refused). Asserted: whenever the checker matches, the reference finds no failing or
undecided verdict and the generator's model does not say the case is wrong. Not vacuous: floors
on matches, refusals per kind and shape, and (IR) on matches among the cases the model says are
right. A refusal is classed by the reference as refuted, undecided (the reference reports
undecided outside its exact class; never counted as agreement) or held.

Skipped unless SENTRYWARD_PTX_NATIVE names a built sentryward-ptx (the IR test: also
sentryward-structural-ir-controls, SENTRYWARD_STRUCTURAL_IR or next to the native).
SENTRYWARD_DIFF_CASES (default 400), SENTRYWARD_DIFF_IR_CASES (default 2000) and
SENTRYWARD_DIFF_SEED (default 0xD263) set size and seed; SENTRYWARD_DIFF_KEEP names a directory
that keeps every PTX case (PTX, input, term, report).
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
IR = os.environ.get("SENTRYWARD_STRUCTURAL_IR") or (
    str(Path(NATIVE).parent / "sentryward-structural-ir-controls") if NATIVE else None)
CASES = int(os.environ.get("SENTRYWARD_DIFF_CASES", "400"))
IR_CASES = int(os.environ.get("SENTRYWARD_DIFF_IR_CASES", "2000"))
SEED = int(os.environ.get("SENTRYWARD_DIFF_SEED", str(0xD263)), 0)
THREADS = 32
ROLE_SETS = [[], [], ["through_transcendental"], ["avoids_transcendental"], ["addend"], ["multiplicand"],
             ["avoids_transcendental", "addend"], ["through_transcendental", "multiplicand"]]
IDENT = [[{"out": 0}, {"out": 1}]]
SECTIONS = ("leaves", "cuts", "bindings")


class Kernel:
    """Straight-line PTX: in CTA c, thread i reads and writes element (c, i) of each f32 object."""

    def __init__(self, name, objects, grid=1):
        self.name, self.objects, self.grid = name, objects, grid
        self.body, self.f, self.h, self.d, self.p, self.r = [], 0, 0, 20, 0, 3
        self.base = {}
        # Per register, the generator's own dataflow model: the source tags some path reaches, those
        # reached through ex2/lg2 above them, and the same two over full paths (no partial mask).
        self.info = {}
        self.stored, self.masks = [], Counter()
        for i, o in enumerate(objects):
            self.body.append(f"ld.param.u64 %rd{i + 1}, [{name}_param_{i}];")
            self.body.append(f"cvta.to.global.u64 %rd{10 + i}, %rd{i + 1};")
            self.base[o] = f"%rd{10 + i}"
        self.body += ["mov.u32 %r1, %tid.x;", "mov.u32 %r2, %ctaid.x;", f"mad.lo.s32 %r3, %r2, {THREADS}, %r1;",
                      "mul.wide.s32 %rd9, %r3, 4;"]

    def freg(self):
        self.f += 1
        return f"%f{self.f}"

    def rreg(self):
        self.r += 1
        return f"%r{self.r}"

    def addr(self, o):
        self.d += 1
        self.body.append(f"add.s64 %rd{self.d}, {self.base[o]}, %rd9;")
        return f"%rd{self.d}"

    def load(self, o):
        a, r = self.addr(o), self.freg()
        self.body.append(f"ld.global.f32 {r}, [{a}];")
        return r

    def store(self, o, r):
        self.stored.append(r)
        self.body.append(f"st.global.f32 [{self.addr(o)}], {r};")

    def op(self, text, n=1):
        rs = [self.freg() for _ in range(n)]
        self.body.append(text.format(*rs))
        return rs[0] if n == 1 else rs

    def warp_sum(self, x, steps):
        """x summed over the lanes a butterfly with these xor offsets reaches."""
        for off in steps:
            a, b, y = self.rreg(), self.rreg(), self.freg()
            self.body += [f"mov.b32 {a}, {x};", f"shfl.sync.bfly.b32 {b}, {a}, {off}, 31, -1;", f"mov.b32 {y}, {b};"]
            s = self.op(f"add.f32 {{0}}, {x}, {y};")
            self.info[s] = self.info[x]
            x = s
        return x

    def text(self):
        params = ",\n".join(f"\t.param .u64 {self.name}_param_{i}" for i in range(len(self.objects)))
        regs = [f".reg .f32 %f<{self.f + 1}>;", f".reg .b16 %h<{self.h + 1}>;", f".reg .pred %p<{self.p + 1}>;",
                f".reg .b32 %r<{self.r + 1}>;", f".reg .b64 %rd<{self.d + 1}>;"]
        body = "\n".join("\t" + x for x in regs + [""] + self.body + ["ret;"])
        return (".version 8.8\n.target sm_90a\n.address_size 64\n\n"
                f".visible .entry {self.name}(\n{params}\n)\n{{\n{body}\n}}\n")


def expression(rng, k, leaves, depth):
    """A random f32 expression over the registers `leaves`, using each at least once."""
    def mk(text, xs, trans=False):
        r = k.op(text)
        alls, thr, full, thr_full = (set().union(*(k.info[x][i] for x in xs)) if xs else set() for i in range(4))
        k.info[r] = (alls, thr | (alls if trans else set()), full, thr_full | (full if trans else set()))
        return r

    def gen(d):
        if d == 0 or rng.random() < 0.25:
            return rng.choice(leaves)
        if rng.random() < 0.1:  # a masked operand: x's bits through one bit operation
            x = gen(d - 1)
            text, keeps = rng.choice(MASKS)
            i, o, t = k.rreg(), k.rreg(), k.rreg()
            k.body += [f"mov.b32 {i}, {x};"] + text.format(o=o, i=i, t=t).split("\n")
            r = k.op(f"mov.b32 {{0}}, {o};")
            alls, thr, full, thr_full = k.info[x]
            k.info[r] = (alls, thr, full, thr_full) if keeps else (alls, thr, set(), set())
            k.masks["full" if keeps else "partial"] += 1
            return r
        c = rng.random()
        if c < 0.2:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"add.f32 {{0}}, {x}, {y};", [x, y])
        if c < 0.38:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"mul.f32 {{0}}, {x}, {y};", [x, y])
        if c < 0.47:
            x, y = gen(d - 1), gen(d - 1)
            return mk(f"fma.rn.f32 {{0}}, {x}, {y}, 0f3F800000;", [x, y])
        if c < 0.55:
            x = gen(d - 1)
            return mk(f"add.f32 {{0}}, {x}, 0f40400000;", [x])
        if c < 0.64:
            x = gen(d - 1)
            return mk(f"ex2.approx.f32 {{0}}, {x};", [x], True)
        if c < 0.69:
            x = gen(d - 1)
            return mk(f"lg2.approx.f32 {{0}}, {x};", [x], True)
        if c < 0.78:  # f16 round trip: a conversion chain
            k.h += 1
            x = gen(d - 1)
            k.body.append(f"cvt.rn.f16.f32 %h{k.h}, {x};")
            return mk(f"cvt.f32.f16 {{0}}, %h{k.h};", [x])
        if c < 0.88:  # a select on a data-dependent condition: the condition's paths count too
            x, y, z = gen(d - 1), gen(d - 1), gen(d - 1)
            k.p += 1
            k.body.append(f"setp.gt.f32 %p{k.p}, {z}, 0f00000000;")
            return mk(f"selp.f32 {{0}}, {x}, {y}, %p{k.p};", [x, y, z])
        # a component of a packed pair
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


def model_holds(k, reg, tag, role):
    """Whether the generator's model says `role` of the argument tagged `tag` holds at `reg`: True,
    False, or None (a role the model does not decide). Reaching (through ex2/lg2) needs a full path;
    avoiding counts partial ones."""
    alls, thr, full, thr_full = k.info[reg]
    if role == "through_transcendental":
        return tag in thr_full
    if role == "avoids_transcendental":
        return tag in full and tag not in thr
    return None


def src(tag):
    return ({tag}, set(), {tag}, set())


# Masks of an f32's bits: (instructions, whether every bit is kept); {t} is a scratch register.
# and 0xffff0000 on an f32's own bits clears its low mantissa in place: a rounding toward zero, full.
# shr 16 used as an f32 is a fake narrowing (the kept bits move; 1.0f gives 0x00003f80): partial.
# An integer product reads every bit only when the other factor is odd: x * (x << 1) drops x's bit
# 31 (round-6 reproducer), x * (x | 1) and x * 3 keep every bit.
MASKS = [("and.b32 {o}, {i}, 65535;", False), ("shl.b32 {o}, {i}, 16;", False), ("shr.u32 {o}, {i}, 24;", False),
         ("prmt.b32 {o}, {i}, 0, 4112;", False), ("bfe.u32 {o}, {i}, 8, 16;", False),
         ("and.b32 {o}, {i}, 2147483647;", False), ("xor.b32 {o}, {i}, 2147483648;", True),
         ("prmt.b32 {o}, {i}, 0, 12816;", True), ("shr.u32 {o}, {i}, 16;", False), ("and.b32 {o}, {i}, 4294901760;", True),
         ("shl.b32 {t}, {i}, 1;\nmul.lo.u32 {o}, {i}, {t};", False), ("mul.lo.u32 {o}, {i}, 3;", True),
         ("or.b32 {t}, {i}, 1;\nmul.lo.u32 {o}, {i}, {t};", True)]


def roles(rng, k, reg, tag, wrong=True):
    """A role set for the argument tagged `tag` of the atom whose value is `reg`: half the time the
    transcendental role the model says holds, a quarter the opposite one (which the checker must
    refuse), else a random set. Without `wrong`: the role that holds, or none (producers, so that a
    case reaches its consumer's roles)."""
    c = rng.random()
    alls, thr = k.info[reg][:2]
    if not wrong:
        if c < 0.6 and tag in alls:
            return ["through_transcendental" if tag in thr else "avoids_transcendental"]
        return []
    if c < 0.75 and tag in alls:
        holds = "through_transcendental" if tag in thr else "avoids_transcendental"
        opposite = "avoids_transcendental" if tag in thr else "through_transcendental"
        return [holds if c < 0.5 else opposite]
    return rng.choice(ROLE_SETS)


def obj(name, rows=1):
    return {"name": name, "element": "f32", "element_bytes": 4, "base_bytes": 0,
            "row_stride_bytes": 4 * THREADS, "rows": rows, "cols": THREADS, "row_indexed": rows > 1}


def opaque(op, args, roles, out):
    return {"op": op, "atom": {"kind": "opaque", "roles": roles}, "format": {"compute": None, "store": "f32"},
            "args": args, "out": out}


def case(rng, name):
    """A random (term, kernel) pair and the model's verdict: False when the case must not match
    (a missing reduction step, or a transcendental role the model says fails), else True."""
    c = rng.random()
    truth = [True]

    def checked(k, reg, tags, role_sets):
        for tag, rs in zip(tags, role_sets):
            for role in rs:
                if model_holds(k, reg, tag, role) is False:
                    truth[0] = False
        return role_sets
    grid = 1
    if c < 0.55:
        shape = "chain"
        grid = rng.choice([1, 1, 4])
        two = rng.random() < 0.5
        objects = ["a", "b", "p"] + (["q"] if two else []) + ["y"]
        k = Kernel(name, objects, grid)
        a, b = k.load("a"), k.load("b")
        k.info[a], k.info[b] = src("a"), src("b")
        p = expression(rng, k, [a], rng.randint(0, 3))
        k.store("p", p)

        def reread(o, r):  # a cut read back through memory or used from its register
            x = k.load(o) if rng.random() < 0.5 else r
            alls, thr, full, thr_full = k.info[r]
            k.info[x] = (alls | {o}, thr, full | {o}, thr_full)
            return x
        nodes = [opaque("P", [{"input": 0, "reads": IDENT}], checked(k, p, ["a"], [roles(rng, k, p, "a", wrong=False)]), "p")]
        regs = {"p": p}
        if two:
            q = expression(rng, k, [reread("p", p)] + ([a] if rng.random() < 0.2 else []), rng.randint(0, 3))
            k.store("q", q)
            regs["q"] = q
            nodes.append(opaque("Q", [{"body": 0, "reads": IDENT}], checked(k, q, ["p"], [roles(rng, k, q, "p", wrong=False)]), "q"))
        pool = [("body", 0, "p")] + ([("body", 1, "q")] if two else []) + [("input", 0, "a"), ("input", 1, "b")]
        if two and rng.random() < 0.5:  # a consumer of both producers: a path from p behind q's cut
            args = [pool[0], pool[1]] + rng.sample(pool[2:], rng.randint(0, 1))
            rng.shuffle(args)
        else:
            args = rng.sample(pool, rng.randint(1, min(3, len(pool))))
        leaves = [reread(n, regs[n]) if kind == "body" else (a if n == "a" else b) for kind, _, n in args]
        y = expression(rng, k, leaves, rng.randint(0, 3))
        k.store("y", y)
        nodes.append(opaque("C", [{kind: ref, "reads": IDENT} for kind, ref, _ in args],
                            checked(k, y, [n for _, _, n in args], [roles(rng, k, y, n) for _, _, n in args]), "y"))
        term = {"objects": [obj(o, grid) for o in objects], "inputs": ["a", "b"], "nodes": nodes}
    elif c < 0.8:
        shape = "unstored"
        uses = rng.randint(1, 2)
        objects = ["a", "b", "c"] + [f"y{i}" for i in range(uses)]
        k = Kernel(name, objects)
        a, b, cc = k.load("a"), k.load("b"), k.load("c")
        k.info[a] = src("a")
        t = expression(rng, k, [a], rng.randint(1, 3))
        p_roles = roles(rng, k, t, "a")
        nodes = [opaque("P", [{"input": 0, "reads": IDENT}], [p_roles], None)]
        for i in range(uses):
            u = t if rng.random() < 0.8 else expression(rng, k, [a], rng.randint(1, 2))
            checked(k, u, ["a"], [p_roles])  # each value a use binds (two values: producer_binding)
            if k.info[u][0] != k.info[u][2]:  # a bound value that reaches a only through a mask
                truth[0] = False
            k.store(f"y{i}", k.op(f"add.f32 {{0}}, {u}, {[b, cc][i]};"))
            nodes.append({"op": "add", "atom": {"kind": "primitive", "operation": "add", "commutative": True},
                          "format": {"compute": None, "store": "f32"},
                          "args": [{"body": 0, "reads": IDENT}, {"input": 1 + i, "reads": IDENT}], "out": f"y{i}"})
        term = {"objects": [obj(o) for o in objects], "inputs": ["a", "b", "c"], "nodes": nodes}
    else:
        # A row reduction through warp shuffles: y[i] = g(sum over lanes of f(a[lane]), b[i]); the
        # term reads all of a's row. A missing butterfly step leaves half the row unread.
        shape = "reduction"
        objects = ["a", "b", "y"]
        k = Kernel(name, objects)
        a, b = k.load("a"), k.load("b")
        k.info[a], k.info[b] = src("a"), src("b")
        v = expression(rng, k, [a], rng.randint(0, 2))
        steps = [16, 8, 4, 2, 1]
        if rng.random() < 0.25:
            steps.remove(rng.choice(steps))
            truth[0] = False
        s = k.warp_sum(v, steps)
        use_b = rng.random() < 0.6
        y = expression(rng, k, [s] + ([b] if use_b else []), rng.randint(0, 2))
        k.store("y", y)
        args = [{"input": 0, "reads": [[{"out": 0}, {"all": THREADS}]]}] + ([{"input": 1, "reads": IDENT}] if use_b else [])
        tags = ["a"] + (["b"] if use_b else [])
        nodes = [opaque("R", args, checked(k, y, tags, [roles(rng, k, y, tg) for tg in tags]), "y")]
        term = {"objects": [obj(o) for o in objects], "inputs": ["a", "b"], "nodes": nodes}
    for r in k.stored:  # a stored value that reaches a source only through a mask
        if r in k.info and k.info[r][0] != k.info[r][2]:
            truth[0] = False
    masked = "partial" if k.masks["partial"] else "full" if k.masks["full"] else None
    term.update({"schema": "sentryward.structural-term/v2", "row": f"diff/{name}", "template": f"diff/{name}"})
    inputs = {"schema": "spverify.kernel-input/v1", "entry": name,
              "launch": {"grid_dim": [grid, 1, 1], "block_dim": [THREADS, 1, 1], "dynamic_shared_bytes": 0},
              "declarations": [{"parameter": i, "type": "pointer", "space": "global", "object": o}
                               for i, o in enumerate(objects)], "arguments": []}
    return shape, grid, truth[0], masked, term, inputs, k.text()


def run_case(native, d, term, inputs, ptx):
    d.mkdir(parents=True, exist_ok=True)
    (d / "k.ptx").write_text(ptx)
    (d / "in.json").write_text(json.dumps(inputs))
    (d / "term.json").write_text(json.dumps(term))
    subprocess.run([native, "--ptx", str(d / "k.ptx"), "--config", str(d / "in.json"), "--analyze-scalar-fold",
                    "--structural-term", str(d / "term.json"), "--structural-report", str(d / "report.json"),
                    "--structural-reference"], capture_output=True, text=True, check=False, timeout=120)
    return json.loads((d / "report.json").read_text())


def reference_problems(ref):
    """The reference's failing or undecided verdicts (roles and each section), or its error."""
    if "error" in ref or "checks" not in ref:
        return [ref.get("error", "no reference")]
    out = [f"roles.{k}" for k in ("fails", "undecided") if ref[k]]
    return out + [f"{s}.{k}" for s in SECTIONS for k in ("fails", "undecided") if ref[s][k]]


def reference_refutes(ref):
    """Whether the reference finds a failing verdict (it confirms a refusal)."""
    return "checks" in ref and bool(ref["fails"] or any(ref[s]["fails"] for s in SECTIONS))


def reference_undecided(ref):
    """The number of the reference's undecided verdicts (roles and each section)."""
    if "checks" not in ref:
        return 0
    return len(ref["undecided"]) + sum(len(ref[s]["undecided"]) for s in SECTIONS)


def refusal_agreement(ref):
    """How the reference reads a case the checker refused: it refutes it, cannot decide some check
    (undecided: neither agreement nor disagreement), or holds everywhere."""
    if reference_refutes(ref):
        return "refusal: reference refutes"
    return "refusal: reference undecided" if reference_undecided(ref) else "refusal: reference holds"


def outcome(r):
    return r["result"] if r["result"] == "matched" else f'{r["result"]}:{r.get("reason")}'


@unittest.skipUnless(NATIVE, "SENTRYWARD_PTX_NATIVE is not set")
class StructuralDifferentialTest(unittest.TestCase):
    def test_checker_never_accepts_where_the_reference_or_the_model_refuses(self):
        rng = random.Random(SEED)
        keep = os.environ.get("SENTRYWARD_DIFF_KEEP")
        outcomes, shapes, agree, violations, masked = Counter(), Counter(), Counter(), [], Counter()
        checks = Counter()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(keep or tmp)
            for i in range(CASES):
                name = f"diff_{i}"
                shape, grid, truth, mask, term, inputs, ptx = case(rng, name)
                r = run_case(NATIVE, root / name, term, inputs, ptx)
                o = outcome(r)
                outcomes[o] += 1
                if mask:
                    masked[(mask, o)] += 1
                shapes[(shape + ("/class" if r.get("class_mode") else ""), "matched" if o == "matched" else "refused")] += 1
                ref = r.get("reference", {})
                if o != "matched":
                    agree[refusal_agreement(ref)] += 1
                    continue
                checks["roles"] += ref.get("checks", 0)
                for s in SECTIONS:
                    checks[s] += ref.get(s, {}).get("checks", 0)
                checks["undecided"] += reference_undecided(ref)
                problems = reference_problems(ref)
                if problems or not truth:
                    violations.append({"case": i, "shape": shape, "reference": problems, "model": truth})
        print(f"\nPTX seed {SEED:#x}, {CASES} cases: {dict(outcomes)}\n  shapes: {dict(shapes)}\n"
              f"  reference checks in matched cases: {dict(checks)}\n  {dict(agree)}\n  masked operands: {dict(masked)}")
        self.assertEqual(violations, [], f"checker matched where the reference or the model refuses (seed {SEED:#x})")
        self.assertGreater(outcomes["matched"], CASES // 10)
        self.assertGreater(outcomes["refused:role"], CASES // 20)
        self.assertGreater(outcomes["refused:leaves"], 0)
        # Masked operands: a case whose model needs a source only a partial mask reaches is never
        # matched (asserted above); some are refused partial_operand (others fail earlier, or reach
        # the source in full elsewhere and match); cases with only full masks still match.
        self.assertGreaterEqual(masked[("partial", "refused:partial_operand")], CASES // 100)
        self.assertGreater(masked[("full", "matched")], 0)
        for s in ("chain/class", "reduction", "unstored"):
            self.assertGreater(shapes[(s, "matched")], 0, s)
        self.assertGreater(checks["roles"], 0)
        self.assertGreater(checks["cuts"], 0)
        self.assertGreater(checks["bindings"], 0)

    @unittest.skipUnless(IR and Path(IR).exists(), "sentryward-structural-ir-controls is not built")
    def test_ir_cases_with_indexed_selections(self):
        out = subprocess.run([IR, "random", str(SEED), str(IR_CASES)], capture_output=True, text=True,
                             check=True, timeout=600).stdout
        rows = [json.loads(x) for x in out.splitlines()]
        self.assertEqual(len(rows), IR_CASES)
        kinds, agree, violations = Counter(), Counter(), []
        for row in rows:
            r = row["report"]
            o = outcome(r)
            kinds[(row["shape"], row["mutation"], o)] += 1
            if o != "matched":
                agree[refusal_agreement(r.get("reference", {}))] += 1
                continue
            problems = reference_problems(r.get("reference", {}))
            if problems or not row["truth"]:
                violations.append({"case": row["case"], "mutation": row["mutation"], "reference": problems})
        print(f"\nIR seed {SEED:#x}, {IR_CASES} cases: {dict(sorted(kinds.items()))}\n  {dict(agree)}")
        self.assertEqual(violations, [], f"checker matched where the reference or the generator refuses (seed {SEED:#x})")
        # Not vacuous: each mutation is refused, and the checker matches nearly every case the
        # generator says is right (an identity that conflates values over-refuses).
        right = [row for row in rows if row["truth"]]
        matched = sum(row["report"]["result"] == "matched" for row in right)
        self.assertGreater(matched, 0.9 * len(right))
        for m in ("index", "recompute", "rebind", "role"):
            self.assertGreater(sum(row["mutation"] == m for row in rows), 0, m)


if __name__ == "__main__":
    unittest.main()
