#include "sentryward/algorithm/checker/footprint_checker.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "sentryward/algorithm/checker/unit_quotient.h"
#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kSamplesPerKind = 4;
constexpr std::size_t kSamplesTotal = 32;

// One declared piece set: rows r < rows of `count` runs, run c of row r being the bytes
// [base + r*stride + c*period, + width), readable and/or writable. Coral's port layout (a port's
// rows, its runs inside a row, its field and self offsets) is one piece set; a pointer bound by
// several ports has several. The legacy `periodic` record is one piece set with base `offset`.
struct Piece {
  std::uint64_t base = 0, rows = 1, stride = 0, period = 0, count = 1, width = 0;
  bool read = true, write = true;
  [[nodiscard]] bool Allows(bool is_read) const { return is_read ? read : write; }
  // Whether [column, column + bytes) lies inside one run.
  [[nodiscard]] bool RunHolds(std::uint64_t column, std::uint64_t bytes) const {
    const auto step = count > 1 ? period : width;
    const auto run = step ? column / step : 0;
    if (run >= count) return false;
    const auto start = run * step;
    return column - start + bytes <= width;
  }
  [[nodiscard]] std::uint64_t RowBytes() const { return (count - 1) * (count > 1 ? period : width) + width; }
  // Whether the piece's bytes are exactly the contiguous range [0, n) for some n >= extent: one run
  // per row and rows back to back (or one row). Then an access contained in [0, extent) is in it.
  [[nodiscard]] bool CoversExtent(std::uint64_t extent) const {
    if (base != 0 || count != 1 || rows == 0) return false;
    if (rows == 1) return width >= extent;
    return stride == width && rows * stride >= extent;
  }
  // [low, low + bytes) inside one run of one row. Rows may overlap (a window such as an indptr's
  // [r, r + 2) at a one-entry stride): every row whose bytes reach `low` is tried.
  [[nodiscard]] bool Contains(std::int64_t low, std::uint64_t bytes) const {
    if (rows == 0 || low < std::int64_t(base)) return false;
    const auto rel = std::uint64_t(low) - base;
    if (rows <= 1 || stride == 0) return RunHolds(rel, bytes);
    const auto row_bytes = RowBytes();
    for (auto r = std::min(rel / stride, rows - 1);; --r) {
      const auto column = rel - r * stride;
      if (column >= row_bytes) break;
      if (RunHolds(column, bytes)) return true;
      if (r == 0) break;
    }
    return false;
  }
  // A symbolic access c + g*(atoms) whose touched hull is [low, high): inside (every valuation
  // lands in one run of some row) when the hull lies inside the rows and g is a multiple of the
  // row stride (the atoms pick a row, the constant fixes the column); outside when the hull misses
  // the rows, or under the same stride condition the column misses every run; else undecided.
  enum class Rows3 { kIn, kOut, kUnknown };
  [[nodiscard]] Rows3 ContainsRows(std::int64_t constant, std::uint64_t gcd, std::int64_t low, std::int64_t high,
                                   std::uint64_t bytes) const {
    const auto top = std::int64_t(base + (rows <= 1 || stride == 0 ? (rows ? (count - 1) * (count > 1 ? period : width) + width : 0)
                                                                   : rows * stride));
    if (rows == 0 || high <= std::int64_t(base) || low >= top) return Rows3::kOut;
    // One row of one run: the piece is the contiguous bytes [base, top), so a hull inside them is in.
    if (rows <= 1 && count <= 1 && low >= std::int64_t(base) && high <= top) return Rows3::kIn;
    if (rows <= 1 || stride == 0 || gcd == 0 || gcd % stride != 0 || stride < RowBytes()) return Rows3::kUnknown;
    if (low < std::int64_t(base) || high > top) return Rows3::kUnknown;
    const auto s = std::int64_t(stride);
    const auto column = ((constant - std::int64_t(base)) % s + s) % s;
    return RunHolds(std::uint64_t(column), bytes) ? Rows3::kIn : Rows3::kOut;
  }
};

struct Declared {
  std::string direction;  // read, write, readwrite
  std::uint64_t extent = 0;
  std::vector<Piece> pieces;  // empty: the whole extent
};

struct ObjectStats {
  std::uint64_t reads = 0, writes = 0, symbolic = 0, undecided = 0, violations = 0, reads_outside = 0;
  std::optional<std::int64_t> min_byte, max_byte;  // [min, max) over bounded accesses
  bool interval_complete = true;                   // false once an access had no bounded interval
};

std::array<std::uint32_t, 3> Unflatten(std::uint32_t linear, const std::array<std::uint32_t, 3>& dim) {
  return {linear % dim[0], (linear / dim[0]) % dim[1], linear / (dim[0] * dim[1])};
}

}  // namespace

// The members' δ of one allocation in one CTA class (class mode): the representative's accesses
// stand for every unit of the class, each moved by its unit's δ. `all` holds 0 for the representative.
struct Shifts {
  std::int64_t low = 0, high = 0;   // min and max δ
  std::vector<std::int64_t> all{0}; // distinct δ values (the representative's 0 included)
  std::uint64_t units = 1;          // class size (the representative's access weight)
};

FootprintReport CheckFootprints(MemoryAccessView accesses, std::span<const ActorInfo> actors,
                                SymbolicAddressModel& model, std::string_view footprints_json,
                                const UnitQuotient* units) {
  FootprintReport report;
  const auto spec = Json::parse(footprints_json);
  const auto dims = [&](const char* key) {
    std::array<std::uint32_t, 3> dim{1, 1, 1};
    if (spec.contains(key))
      for (std::size_t i = 0; i < 3 && i < spec[key].size(); ++i) dim[i] = spec[key][i].get<std::uint32_t>();
    return dim;
  };
  const auto grid = dims("grid_dim"), block = dims("block_dim");
  std::map<std::string, Declared> declared;
  for (const auto& item : spec.at("objects")) {
    Declared decl{item.at("direction").get<std::string>(), item.at("extent_bytes").get<std::uint64_t>(), {}};
    const auto& dir = decl.direction;
    if (item.contains("periodic")) {
      const auto& q = item.at("periodic");
      const std::uint64_t stride = q.at("stride"), offset = q.at("offset");
      decl.pieces.push_back(Piece{offset, decl.extent > offset ? (decl.extent - offset + stride - 1) / stride : 0, stride,
                                  q.at("period"), q.at("count"), q.at("width"), dir != "write", dir != "read"});
    }
    for (const auto& q : item.value("pieces", Json::array())) {
      const auto pdir = q.value("direction", dir);
      decl.pieces.push_back(Piece{q.value("base", std::uint64_t(0)), q.value("rows", std::uint64_t(1)),
                                  q.value("stride", std::uint64_t(0)), q.value("period", std::uint64_t(0)),
                                  q.value("count", std::uint64_t(1)), q.at("width"), pdir != "write", pdir != "read"});
    }
    declared[item.at("object").get<std::string>()] = std::move(decl);
  }
  // Block and in-block thread ordinal of each actor (actor ids of one CTA are consecutive in thread order).
  ActorCtas ctas(actors);
  std::unordered_map<std::uint32_t, std::uint32_t> thread_of;
  std::map<std::uint32_t, std::uint32_t> next_in_cta;
  for (const auto& actor : actors) thread_of[actor.id.value()] = next_in_cta[actor.cta.value()]++;

  // Class mode: the δ of `object` over the class of the representative CTA `cta`. A piecewise δ
  // (sites moving by different amounts) or a missing δ restarts the launch per unit.
  std::map<std::pair<std::uint32_t, std::string>, Shifts> shift_cache;
  const auto shifts = [&](std::uint32_t cta, const std::string& object) -> const Shifts& {
    static const Shifts none;
    if (!units || cta == ActorCtas::kNoCta) return none;
    const auto klass = units->units.at(units->UnitOf(cta)).klass;
    auto [slot, fresh] = shift_cache.try_emplace({klass, object});
    if (!fresh) return slot->second;
    const auto& c = units->classes.at(klass);
    auto& out = slot->second;
    out.units = c.units.size();
    std::set<std::int64_t> seen{0};
    if (c.units.size() > 1 && c.pieces.contains(object)) throw UnitClassRestart{"footprint_piecewise_delta:" + object};
    for (std::size_t i = 1; i < c.units.size(); ++i) {
      if (i >= c.delta.size()) throw UnitClassRestart{"footprint_delta_missing:" + object};
      const auto found = c.delta[i].find(object);
      if (found == c.delta[i].end()) throw UnitClassRestart{"footprint_delta_missing:" + object};
      seen.insert(found->second);
    }
    out.all.assign(seen.begin(), seen.end());
    out.low = out.all.front();
    out.high = out.all.back();
    return out;
  };

  std::map<std::string, ObjectStats> stats;
  std::map<std::tuple<std::string, std::string, std::string>, std::size_t> kind_counts;
  Json violations = Json::array(), undecided = Json::array();
  std::map<std::string, std::size_t> undecided_reasons;

  const auto sample = [&](const MemoryAccess& access, const std::string& kind, const std::string& object,
                          const ContainmentVerdict& v, std::optional<std::uint64_t> extent) {
    Json item{{"kind", kind}, {"object", object}, {"mode", access.mode == AccessMode::kRead ? "read" : "write"},
              {"bytes", access.bytes ? Json(*access.bytes) : Json(nullptr)},
              {"extent_bytes", extent ? Json(*extent) : Json(nullptr)}};
    const auto cta = ctas(access.actor);
    if (cta != ActorCtas::kNoCta) {
      item["cta"] = cta;
      item["block"] = Unflatten(cta, grid);
      const auto thread = thread_of.count(access.actor.value()) ? thread_of.at(access.actor.value()) : 0;
      item["thread"] = thread;
      item["thread_idx"] = Unflatten(thread, block);
    }
    item["actor"] = access.actor.value();
    item["location"] = {{"file", access.location.file}, {"line", access.location.line}};
    if (v.status == ContainmentVerdict::Status::kViolated) {
      item["side"] = v.side;
      item["witness"] = v.witness;
      item["offset_kind"] = v.constant_offset ? "constant" : "symbolic";
    } else if (!v.reason.empty()) {
      item["reason"] = v.reason;
    }
    return item;
  };
  // Coral's read clause is non-interference (outputs depend only on the declared input bytes), not
  // read containment: under "noninterference" a read outside the declared bytes is reported (the
  // read half is not shown contained) but is not a violation, since a masked tiling may load bytes
  // it never uses. Writes are always checked by containment.
  std::size_t current_weight = 1;  // the class size of the access being checked (class mode)
  const bool reads_noninterference = spec.value("reads", std::string("containment")) == "noninterference";
  std::size_t write_violations = 0, read_violations = 0, write_unknown = 0, read_unknown = 0, reads_outside = 0;
  Json outside_reads = Json::array();
  const auto violation = [&](const MemoryAccess& access, const std::string& kind, const std::string& object,
                             const ContainmentVerdict& v, std::optional<std::uint64_t> extent) {
    const bool read = access.mode == AccessMode::kRead;
    const std::size_t weight = current_weight;
    auto& count = kind_counts[{kind, object, read ? "read" : "write"}];
    if (read && reads_noninterference) {
      reads_outside += weight;
      stats[object].reads_outside += weight;
      if (count++ < kSamplesPerKind && outside_reads.size() < kSamplesTotal)
        outside_reads.push_back(sample(access, kind, object, v, extent));
      return;
    }
    report.violations += weight;
    (read ? read_violations : write_violations) += weight;
    stats[object].violations += weight;
    if (count++ < kSamplesPerKind && violations.size() < kSamplesTotal)
      violations.push_back(sample(access, kind, object, v, extent));
  };
  const auto undecided_at = [&](const MemoryAccess& access, const std::string& object, const ContainmentVerdict& v,
                                std::uint64_t extent) {
    report.unknown += current_weight;
    (access.mode == AccessMode::kRead ? read_unknown : write_unknown) += current_weight;
    stats[object].undecided += current_weight;
    if (undecided_reasons[v.reason]++ == 0 && undecided.size() < kSamplesTotal)
      undecided.push_back(sample(access, "undecided", object, v, extent));
  };

  for (const auto& access : accesses) {
    // Only a known zero width has no bytes to check; an unknown width is classified below.
    if ((access.bytes && *access.bytes == 0) || !std::holds_alternative<AddressId>(access.address)) continue;
    // The verdict without an extent names the allocation and space; global memory only.
    auto probe = model.containment(access, std::nullopt);
    // A probe that could not classify the access (no space: model error, unproved invariants,
    // missing binding) is undecided, never skipped.
    if (probe.address_space.empty() && probe.status == ContainmentVerdict::Status::kUnknown) {
      ++report.checked;
      current_weight = 1;
      undecided_at(access, "<unclassified>", probe, 0);
      continue;
    }
    if (probe.address_space != "global") continue;
    if (!access.bytes) {
      // A global access of unknown width: its bytes cannot be checked (undecided, never a pass).
      ++report.checked;
      current_weight = 1;
      undecided_at(access, probe.allocation.empty() ? "<unresolved>" : probe.allocation, probe, 0);
      continue;
    }
    ++report.checked;
    const std::string object = probe.allocation.empty() ? "<unresolved>" : probe.allocation;
    const Shifts unshifted;
    const auto& shift = probe.allocation.empty() ? unshifted : shifts(ctas(access.actor), object);
    current_weight = shift.units;
    auto& st = stats[object];
    (access.mode == AccessMode::kRead ? st.reads : st.writes) += shift.units;
    const auto found = declared.find(object);
    if (found == declared.end()) {
      violation(access, "undeclared_object", object, probe, std::nullopt);
      continue;
    }
    const auto& decl = found->second;
    if (access.mode == AccessMode::kWrite && decl.direction == "read")
      violation(access, "write_to_read_only", object, probe, decl.extent);
    if (access.mode == AccessMode::kRead && decl.direction == "write")
      violation(access, "read_from_write_only", object, probe, decl.extent);
    // Every member's range [o + δ, o + δ + w) inside [0, extent): the representative's in
    // [-min δ, extent - max δ). A symbolic offset in a shared class needs atoms bounded per object
    // (the member reads another element of the same table, under the same bounds); otherwise the
    // launch is analysed per unit.
    const std::int64_t lower = -shift.low;
    const std::uint64_t upper = shift.high > 0 ? (decl.extent > std::uint64_t(shift.high) ? decl.extent - shift.high : 0)
                                               : decl.extent;
    const auto verdict = model.containment(access, upper, 2000, lower);
    if (shift.units > 1 && !verdict.constant_offset && (!verdict.min_byte || verdict.field_bounded))
      throw UnitClassRestart{"footprint_symbolic_member:" + object};
    if (!verdict.constant_offset) st.symbolic += shift.units;
    if (verdict.min_byte && verdict.max_byte_exclusive) {
      const auto lo = *verdict.min_byte + shift.low, hi = *verdict.max_byte_exclusive + shift.high;
      st.min_byte = st.min_byte ? std::min(*st.min_byte, lo) : lo;
      st.max_byte = st.max_byte ? std::max(*st.max_byte, hi) : hi;
    } else {
      st.interval_complete = false;
    }
    if (!decl.pieces.empty() && verdict.status == ContainmentVerdict::Status::kContained) {
      // Declared pieces: the bytes must lie in one run of one row of a piece set whose direction
      // allows the access, for the representative and every class member (moved by its δ).
      // Exact for a constant offset; a symbolic one is decided when its atoms bound a hull and
      // their coefficients are multiples of a piece set's row stride. Otherwise undecided.
      const bool is_read = access.mode == AccessMode::kRead;
      std::optional<std::string> failure;  // kind of the first member outside
      std::int64_t failed_at = 0, failed_delta = 0;
      bool undecided_here = false;
      for (const auto d : shift.all) {
        bool allowed = false, held = false, decided = true;
        for (const auto& piece : decl.pieces) {
          bool in = false;
          // The piece covers [0, extent), and the access cannot straddle two of its rows: one row,
          // or a symbolic offset whose step is a multiple of the row stride and whose bytes fit in
          // the rest of the row from its column (the declared contract is one run of one row).
          const auto stride = piece.stride ? std::int64_t(piece.stride) : std::int64_t(1);
          const bool one_row_access =
              piece.rows == 1 ||
              (!verdict.constant_offset && verdict.constant && verdict.coefficient_gcd % piece.stride == 0 &&
               std::uint64_t((*verdict.constant % stride + stride) % stride) + *access.bytes <= piece.stride);
          if (d == 0 && piece.CoversExtent(decl.extent) && one_row_access) {
            in = true;  // contained in [0, extent), which the piece covers, within one row
          } else if (verdict.constant_offset) {
            in = piece.Contains(*verdict.min_byte + d, *access.bytes);
          } else if (verdict.constant && verdict.min_byte && verdict.max_byte_exclusive) {
            const auto r = piece.ContainsRows(*verdict.constant + d, verdict.coefficient_gcd, *verdict.min_byte + d,
                                              *verdict.max_byte_exclusive + d, *access.bytes);
            in = r == Piece::Rows3::kIn;
            if (r == Piece::Rows3::kUnknown) decided = false;
          } else {
            decided = false;
          }
          held = held || in;
          if (in && piece.Allows(is_read)) { allowed = true; break; }
        }
        if (allowed) continue;
        if (!decided && !held) { undecided_here = true; break; }
        failure = held ? (is_read ? "read_from_write_only" : "write_to_read_only") : "outside_declared_pieces";
        failed_at = (verdict.constant_offset ? *verdict.min_byte : *verdict.constant) + d;
        failed_delta = d;
        break;
      }
      if (undecided_here) {
        auto undecided_verdict = verdict;
        undecided_verdict.status = ContainmentVerdict::Status::kUnknown;
        undecided_verdict.reason = "declared pieces: symbolic offset not decided (no atom hull, or coefficients not a multiple of a row stride)";
        undecided_at(access, object, undecided_verdict, decl.extent);
      } else if (failure) {
        auto outside = verdict;
        outside.status = ContainmentVerdict::Status::kViolated;
        outside.side = *failure;
        outside.witness = "offset=" + std::to_string(failed_at) +
                          (failed_delta ? " (class member, delta " + std::to_string(failed_delta) + ")" : "");
        violation(access, *failure, object, outside, decl.extent);
      }
      continue;
    }
    switch (verdict.status) {
      case ContainmentVerdict::Status::kContained: break;
      case ContainmentVerdict::Status::kViolated:
        violation(access, "out_of_bounds", object, verdict, decl.extent);
        break;
      case ContainmentVerdict::Status::kUnknown:
        undecided_at(access, object, verdict, decl.extent);
        break;
    }
  }

  Json objects = Json::array();
  for (const auto& [name, st] : stats) {
    Json entry{{"object", name}, {"reads", st.reads}, {"writes", st.writes}, {"symbolic_accesses", st.symbolic},
               {"undecided_accesses", st.undecided}, {"violations", st.violations},
               {"reads_outside_declared", st.reads_outside}};
    if (const auto it = declared.find(name); it != declared.end()) {
      entry["declared_direction"] = it->second.direction;
      entry["declared_extent_bytes"] = it->second.extent;
      if (!it->second.pieces.empty()) entry["declared_piece_sets"] = it->second.pieces.size();
    }
    entry["inferred_direction"] = st.reads && st.writes ? "readwrite" : st.writes ? "write" : "read";
    if (st.min_byte && st.max_byte) {
      entry["inferred_min_byte"] = *st.min_byte;
      entry["inferred_max_byte_exclusive"] = *st.max_byte;
      entry["inferred_interval_complete"] = st.interval_complete;
      if (const auto it = declared.find(name); it != declared.end() && st.interval_complete && st.violations == 0)
        entry["unused_declared_bytes_outside_hull"] =
            std::int64_t(it->second.extent) - (*st.max_byte - *st.min_byte);
    }
    objects.push_back(std::move(entry));
  }
  for (const auto& [name, decl] : declared)
    if (!stats.contains(name))
      objects.push_back({{"object", name}, {"reads", 0}, {"writes", 0}, {"declared_direction", decl.direction},
                         {"declared_extent_bytes", decl.extent}, {"note", "no global access"}});

  report.status = report.violations ? FootprintStatus::kViolated
                  : report.unknown  ? FootprintStatus::kInconclusive
                                    : FootprintStatus::kContained;
  Json out{{"status", report.status == FootprintStatus::kViolated       ? "violated"
                      : report.status == FootprintStatus::kInconclusive ? "inconclusive"
                                                                        : "contained"},
           {"checked_accesses", report.checked}, {"violation_count", report.violations},
           {"undecided_count", report.unknown}, {"objects", std::move(objects)},
           {"violations", std::move(violations)}, {"violation_samples_truncated", report.violations > 0 && report.violations > kSamplesTotal},
           {"undecided", std::move(undecided)},
           // The two halves Coral consumes: writes by containment (so bytes outside the declared
           // pieces are unchanged); reads by containment, or under "noninterference" reported only.
           {"read_rule", reads_noninterference ? "noninterference" : "containment"},
           {"writes", write_violations ? "violated" : write_unknown ? "inconclusive" : "contained"},
           {"reads", read_violations ? "violated" : reads_outside ? "not_contained" : read_unknown ? "inconclusive" : "contained"},
           {"reads_outside_declared", reads_outside}, {"outside_reads", std::move(outside_reads)},
           {"scope", "all CTAs of the stated launch (class representatives moved by each member's delta in class mode); trusted: declared extents, directions and tensor_properties"}};
  report.json = out.dump();
  return report;
}

}  // namespace sentryward::algorithm
