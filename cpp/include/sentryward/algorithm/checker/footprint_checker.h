#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/symbolic_address_model.h"
#include "sentryward/algorithm/source_ir/program.h"

namespace sentryward::algorithm {

enum class FootprintStatus {
  kContained,      // every global access is inside its object's declared extent and direction
  kViolated,       // some access is outside, writes a read-only object, or reads a write-only one
  kInconclusive,   // no violation found, but some access could not be decided
};

struct FootprintReport final {
  FootprintStatus status = FootprintStatus::kContained;
  std::size_t violations = 0;
  std::size_t unknown = 0;
  std::size_t checked = 0;
  // The "footprint" section of the result: per-object declared and inferred footprint, counts,
  // and violation / undecided samples (with block, thread, PTX line, offset, extent, witness).
  std::string json;
};

// Checks every global memory access against the footprint declarations (`footprints_json`:
// {"grid_dim","block_dim","objects":[{"object","direction","extent_bytes"}]}) with the
// symbolic address model, and infers the touched interval of each object.
[[nodiscard]] FootprintReport CheckFootprints(MemoryAccessView accesses,
                                              std::span<const ActorInfo> actors,
                                              SymbolicAddressModel& model,
                                              std::string_view footprints_json);

}  // namespace sentryward::algorithm
