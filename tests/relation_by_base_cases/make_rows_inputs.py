"""Writes the distinct_rows control inputs (docs/design/distinct-rows.md) from base_grid2.json:
two blocks (requests), src_tab in [0, 3], dst_tab in [0, 7], entry_units 4. Run from this
directory; the inputs are committed, this script records how they were made."""
import copy
import json
from pathlib import Path

INPUTS = Path(__file__).resolve().parent / "inputs"
BASE = json.loads((INPUTS / "base_grid2.json").read_text())


def field(obj, offset, stride, count, **extra):
    return {"object": obj, "offset": offset, "stride": stride, "count": count, **extra}


def relation(write, read, kind="distinct_rows"):
    return {"kind": kind, "write": write, "read": read}


SRC = field("src_tab", 0, 8, 2)
CASES = {
    # Block s: write row s of dst_tab (stride 8), read row s of src_tab.
    "rows_grid2": [relation(field("dst_tab", 0, 8, 2), SRC)],
    # k_gdn's next[s * positions] with positions = 2: rows at byte 16 s.
    "rows_positions": [relation(field("dst_tab", 0, 16, 2), SRC)],
    # 64 rows, for rb9's reads of src_tab[(t + 2) & 63].
    "rows_grid2_count64": [relation(field("dst_tab", 0, 8, 64), field("src_tab", 0, 8, 64))],
    # Declines (no separation).
    "rows_offset_off_stride": [relation(field("dst_tab", 8, 16, 2), SRC)],
    "rows_wrong_stride": [relation(field("dst_tab", 0, 24, 2), SRC)],
    "rows_count1": [relation(field("dst_tab", 0, 8, 1), field("src_tab", 0, 8, 1))],
    "rows_read_other_object": [relation(field("dst_tab", 0, 8, 2), field("dst_tab", 0, 8, 2))],
    # Refused inputs (malformed).
    "bad_bits": [relation(field("dst_tab", 0, 8, 2, bits=32), SRC)],
    "bad_offset": [relation(field("dst_tab", 4, 8, 2), SRC)],
    "bad_stride": [relation(field("dst_tab", 0, 12, 2), SRC)],
    "bad_stride_zero": [relation(field("dst_tab", 0, 0, 2), SRC)],
    "bad_count_zero": [relation(field("dst_tab", 0, 8, 0), field("src_tab", 0, 8, 0))],
    "bad_counts_differ": [relation(field("dst_tab", 0, 8, 2), field("src_tab", 0, 8, 3))],
    "bad_out_of_range": [relation(field("dst_tab", 2**63 - 8, 8, 2), SRC)],
    "bad_kind": [relation(field("dst_tab", 0, 8, 2), SRC, kind="distinct")],
    "bad_unknown_object": [relation(field("next", 0, 8, 2), SRC)],
    "bad_no_properties": [relation(field("state", 0, 8, 2), SRC)],
}

for name, relations in CASES.items():
    data = copy.deepcopy(BASE)
    data["field_relations"] = relations
    (INPUTS / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")
# A 32-bit table: its tensor_properties are s32, so its lookups are 4-byte words.
data = copy.deepcopy(BASE)
for p in data["tensor_properties"]:
    if p["object"] == "dst_tab":
        p["element_type"] = "s32"
data["field_relations"] = CASES["rows_grid2"]
(INPUTS / "bad_s32_table.json").write_text(json.dumps(data, indent=1) + "\n")
