"""Write inputs/loop_*.json for src/loop.cu (stage B cases). Run: python3 src/loop_inputs.py"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[1]


def piece(n, direction):
    return {"base": 0, "rows": 1, "stride": n, "period": n, "count": 1, "width": n, "direction": direction}


def obj(name, n, direction):
    return {"object": name, "direction": direction, "extent_bytes": n, "pieces": [piece(n, direction)]}


def premise(name, offset, lo, hi, count=1):
    return [{"object": name, "element_type": "u32", "property": p, "value": v,
             "field": {"offset": offset, "stride": 4, "count": count}}
            for p, v in (("lower_bound", lo), ("upper_bound", hi))]


def inp(entry, names, footprints, props, grid=1):
    return {
        "schema": "spverify.kernel-input/v1", "entry": entry,
        "launch": {"grid_dim": [grid, 1, 1], "block_dim": [32, 1, 1], "dynamic_shared_bytes": 0},
        "declarations": [{"parameter": k, "type": "pointer", "space": "global", "object": n} for k, n in enumerate(names)],
        "arguments": [], "tensor_properties": props, "footprints": footprints,
        "footprint_reads": "containment",
    }


# One warp; every lane runs the same loop (the footprint is the check).
N = 64  # elements of in/out
CASES = {
    # n = hdr[0] in [0, 64]: every i < n is inside 64 elements.
    "loop_copy_n": inp("k_copy_n", ["hdr", "in", "out"],
                       [obj("hdr", 4, "read"), obj("in", 4 * N, "read"), obj("out", 4 * N, "write")],
                       premise("hdr", 0, 0, N)),
    "loop_copy_unrolled": inp("k_copy_unrolled", ["hdr", "in", "out"],
                              [obj("hdr", 4, "read"), obj("in", 4 * N, "read"), obj("out", 4 * N, "write")],
                              premise("hdr", 0, 0, N)),
    # Control: i <= n reaches element 64.
    "loop_copy_le": inp("k_copy_le", ["hdr", "in", "out"],
                        [obj("hdr", 4, "read"), obj("in", 4 * N, "read"), obj("out", 4 * N, "write")],
                        premise("hdr", 0, 0, N)),
    # Control: the read index is a carried value loaded from data (havocked): not contained.
    "loop_chase": inp("k_chase", ["hdr", "idx", "in", "out"],
                      [obj("hdr", 4, "read"), obj("idx", 4 * N, "read"), obj("in", 4 * N, "read"),
                       obj("out", 4 * N, "write")],
                      premise("hdr", 0, 0, N)),
    # k_conv's shape: 4 CTAs, rows [ends[s-1], ends[s]) with ends in [0, 64].
    "loop_rows": inp("k_rows", ["ends", "in", "out"],
                     [obj("ends", 16, "read"), obj("in", 4 * N, "read"), obj("out", 4 * N, "write")],
                     premise("ends", 0, 0, N, count=4), grid=4),
}
for name, d in CASES.items():
    (HERE / "inputs" / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
