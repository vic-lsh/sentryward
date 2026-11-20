"""Write inputs/path_*.json for src/path.cu (stage A cases). Run: python3 src/path_inputs.py"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parents[1]
PTX = (HERE / "ptx" / "path.ptx").read_text()


def branch_line(entry):
    """The PTX line of the entry's one conditional branch."""
    lines = PTX.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(f".visible .entry {entry}("))
    return next(i + 1 for i in range(start, len(lines)) if re.match(r"\s*@%p\d+ bra", lines[i]))


def piece(n, direction):
    return {"base": 0, "rows": 1, "stride": n, "period": n, "count": 1, "width": n, "direction": direction}


def inp(entry, out_bytes, taken=None, premise=None, z=None):
    d = {
        "schema": "spverify.kernel-input/v1", "entry": entry,
        "launch": {"grid_dim": [1, 1, 1], "block_dim": [32, 1, 1], "dynamic_shared_bytes": 0},
        "declarations": [{"parameter": 0, "type": "pointer", "space": "global", "object": "hdr"},
                         {"parameter": 1, "type": "pointer", "space": "global", "object": "out"}],
        "arguments": [] if z is None else [{"parameter": 2, "type": "u32", "value": z}],
        "footprints": [{"object": "hdr", "direction": "read", "extent_bytes": 8, "pieces": [piece(8, "read")]},
                       {"object": "out", "direction": "write", "extent_bytes": out_bytes, "pieces": [piece(out_bytes, "write")]}],
        "footprint_reads": "noninterference",
    }
    if taken is not None:
        d["branch_decisions"] = [{"line": branch_line(entry), "taken": taken}]
    if premise is not None:
        lo, hi = premise
        d["tensor_properties"] = [{"object": "hdr", "element_type": "u32", "property": p, "value": v,
                                   "field": {"offset": 0, "stride": 4, "count": 1}}
                                  for p, v in (("lower_bound", lo), ("upper_bound", hi))]
    return d


CASES = {
    # The exit branch is taken when the guard fails; the access is on the not-taken path.
    "path_eq": inp("k_eq", 32, taken=False),
    "path_eq_past": inp("k_eq_past", 32, taken=False),
    "path_two": inp("k_two", 32, taken=False),
    "path_two_past": inp("k_two_past", 32, taken=False),
    "path_or": inp("k_or", 32, taken=False),
    "path_and": inp("k_and", 32, taken=False),
    "path_or_known": inp("k_or_known", 32, taken=False, z=0),
    "path_or_known_past": inp("k_or_known_past", 32, taken=False, z=0),
    "path_mask": inp("k_mask", 64, premise=(0, 15)),
    "path_mask_odd": inp("k_mask_odd", 64, premise=(0, 63)),
}
for name, d in CASES.items():
    (HERE / "inputs" / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
