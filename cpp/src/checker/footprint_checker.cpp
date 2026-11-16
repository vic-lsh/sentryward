#include "sentryward/algorithm/checker/footprint_checker.h"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "sentryward/algorithm/memory/family_access.h"

namespace sentryward::algorithm {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kSamplesPerKind = 4;
constexpr std::size_t kSamplesTotal = 32;

struct Declared {
  std::string direction;  // read, write, readwrite
  std::uint64_t extent = 0;
};

struct ObjectStats {
  std::uint64_t reads = 0, writes = 0, symbolic = 0, undecided = 0, violations = 0;
  std::optional<std::int64_t> min_byte, max_byte;  // [min, max) over bounded accesses
  bool interval_complete = true;                   // false once an access had no bounded interval
};

std::array<std::uint32_t, 3> Unflatten(std::uint32_t linear, const std::array<std::uint32_t, 3>& dim) {
  return {linear % dim[0], (linear / dim[0]) % dim[1], linear / (dim[0] * dim[1])};
}

}  // namespace

FootprintReport CheckFootprints(MemoryAccessView accesses, std::span<const ActorInfo> actors,
                                SymbolicAddressModel& model, std::string_view footprints_json) {
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
  for (const auto& item : spec.at("objects"))
    declared[item.at("object").get<std::string>()] = {item.at("direction").get<std::string>(),
                                                      item.at("extent_bytes").get<std::uint64_t>()};
  // Block and in-block thread ordinal of each actor (actor ids of one CTA are consecutive in thread order).
  ActorCtas ctas(actors);
  std::unordered_map<std::uint32_t, std::uint32_t> thread_of;
  std::map<std::uint32_t, std::uint32_t> next_in_cta;
  for (const auto& actor : actors) thread_of[actor.id.value()] = next_in_cta[actor.cta.value()]++;

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
  const auto violation = [&](const MemoryAccess& access, const std::string& kind, const std::string& object,
                             const ContainmentVerdict& v, std::optional<std::uint64_t> extent) {
    ++report.violations;
    ++stats[object].violations;
    auto& count = kind_counts[{kind, object, access.mode == AccessMode::kRead ? "read" : "write"}];
    if (count++ < kSamplesPerKind && violations.size() < kSamplesTotal)
      violations.push_back(sample(access, kind, object, v, extent));
  };

  for (const auto& access : accesses) {
    if (access.bytes.value_or(0) == 0 || !std::holds_alternative<AddressId>(access.address)) continue;
    // The verdict without an extent names the allocation and space; global memory only.
    auto probe = model.containment(access, std::nullopt);
    if (probe.address_space != "global") continue;
    ++report.checked;
    const std::string object = probe.allocation.empty() ? "<unresolved>" : probe.allocation;
    auto& st = stats[object];
    (access.mode == AccessMode::kRead ? st.reads : st.writes)++;
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
    const auto verdict = model.containment(access, decl.extent);
    if (!verdict.constant_offset) ++st.symbolic;
    if (verdict.min_byte && verdict.max_byte_exclusive) {
      st.min_byte = st.min_byte ? std::min(*st.min_byte, *verdict.min_byte) : *verdict.min_byte;
      st.max_byte = st.max_byte ? std::max(*st.max_byte, *verdict.max_byte_exclusive) : *verdict.max_byte_exclusive;
    } else {
      st.interval_complete = false;
    }
    switch (verdict.status) {
      case ContainmentVerdict::Status::kContained: break;
      case ContainmentVerdict::Status::kViolated:
        violation(access, "out_of_bounds", object, verdict, decl.extent);
        break;
      case ContainmentVerdict::Status::kUnknown:
        ++report.unknown;
        ++st.undecided;
        if (undecided_reasons[verdict.reason]++ == 0 && undecided.size() < kSamplesTotal)
          undecided.push_back(sample(access, "undecided", object, verdict, decl.extent));
        break;
    }
  }

  Json objects = Json::array();
  for (const auto& [name, st] : stats) {
    Json entry{{"object", name}, {"reads", st.reads}, {"writes", st.writes}, {"symbolic_accesses", st.symbolic},
               {"undecided_accesses", st.undecided}, {"violations", st.violations}};
    if (const auto it = declared.find(name); it != declared.end()) {
      entry["declared_direction"] = it->second.direction;
      entry["declared_extent_bytes"] = it->second.extent;
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
           {"scope", "all CTAs of the stated launch; trusted: declared extents, directions and tensor_properties"}};
  report.json = out.dump();
  return report;
}

}  // namespace sentryward::algorithm
