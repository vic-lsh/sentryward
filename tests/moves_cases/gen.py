"""Writes the moves-mode fixtures (inputs/*.json, terms/*.json) for tests/test_moves_native.py.

ptx/gather.ptx is Coral's k_gather (native/gather.cu: for each metadata row r whose emit word
`meta[8 + 4r + 3]` is s >= 0, hs row s is hn row r), compiled by nvcc 13.1.80 (-O3 --ptx,
sm_90a) with HID 2048, then edited to HID 64 (three literals) so a run takes about a second.
The other ptx/*.ptx files are hand edits of it, one change each (named in the test).

Run from this directory: python3 gen.py
"""
import json

R, H = 2, 64      # rows (CTAs) and row elements (bf16)
ROW = 2 * H       # row bytes
META = 4 * (8 + 4 * R)


def bound(prop, value, offset, stride, count):
    return {"object": "meta", "element_type": "s32", "property": prop, "value": value,
            "field": {"offset": offset, "stride": stride, "count": count}}


def kernel_input(lo, hi, injective, block=32):
    tp = [bound("lower_bound", R, 0, 4, 1), bound("upper_bound", R, 0, 4, 1),
          bound("lower_bound", lo, 44, 16, R), bound("upper_bound", hi, 44, 16, R)]
    if injective:
        tp.append({"object": "meta", "element_type": "s32", "property": "injective"})
    return {"schema": "spverify.kernel-input/v1", "entry": "k_gather",
            "launch": {"grid_dim": [R, 1, 1], "block_dim": [block, 1, 1], "dynamic_shared_bytes": 0},
            "declarations": [{"parameter": 0, "type": "pointer", "space": "global", "object": "meta"},
                             {"parameter": 1, "type": "pointer", "space": "global", "object": "hn"},
                             {"parameter": 2, "type": "pointer", "space": "global", "object": "hs"}],
            "arguments": [], "tensor_properties": tp}


def term(live, src_row=0, unit=ROW, dst_rows=R, none=-1):
    return {"schema": "sentryward.moves-term/v1", "row": "sm121/k_gather",
            "unit_bytes": unit,
            "source": {"object": "hn", "base_bytes": src_row * ROW, "unit_stride_bytes": ROW, "units": R},
            "destination": {"object": "hs", "base_bytes": 0, "unit_stride_bytes": ROW, "units": dst_rows,
                            "index": {"object": "meta", "element": "s32", "offset_bytes": 44,
                                      "stride_bytes": 16, "none": none}},
            "live": live}


def dump(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=1)
        f.write("\n")


# Live case: every emit word in [0, R); injective (the two rows never name one destination row).
dump("inputs/live.json", kernel_input(0, R - 1, True))
# None case: every emit word is the excluded value -1.
dump("inputs/none.json", kernel_input(-1, -1, False))
dump("terms/live.json", term("all"))
dump("terms/none.json", term("none"))
# Mismatch controls on the term side.
dump("terms/live_src_shift.json", term("all", src_row=1))   # claims source row r + 1
dump("terms/live_half_unit.json", term("all", unit=ROW // 2))  # claims half a row per unit
