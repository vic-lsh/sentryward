#pragma once
// Static facts that make explicit branch choices (`branch_decisions`) usable as a case split.
//
// A run with a choice at line L covers the executions whose every lane takes that outcome at L.
// An enumeration of choices (one run per outcome, recursively) covers every execution when
//   (1) each chosen branch is reached at most once per lane: its block is in no CFG cycle
//       (`in_loop` false), so a lane's path is a sequence of distinct choices; and
//   (2) no lane's accesses depend on another lane's path: the kernel has no cross-lane operation
//       (shared memory, atomics, barriers, warp shuffles/votes/reductions) and no call whose body
//       is not analysed here.
// These facts are syntactic over the PTX function (no execution); they over-approximate: a block in
// a cycle may still run once, and a listed operation may not move data between lanes.
#include "sentryward/frontend/ptx/module.h"

#include <nlohmann/json.hpp>
#include <set>

namespace sentryward::frontend::ptx {
// {"lines": [{"line", "in_loop", "branch"}], "cross_lane_ops": {opcode class: count}, "calls": n}
// for the PTX lines `lines` of `function`; `branch` is false when no conditional branch or
// predicated exit is on that line (a choice there decides nothing).
nlohmann::json ChoiceFacts(const Function &function, const std::set<std::uint32_t> &lines);
}  // namespace sentryward::frontend::ptx
