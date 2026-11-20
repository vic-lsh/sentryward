#include "sentryward/frontend/ptx/unit_classes.h"

#include "sentryward/frontend/ptx/unit_facts.h"

#include <cctype>
#include <cstring>
#include <map>
#include <set>
#include <type_traits>
#include <utility>

namespace sentryward::frontend::ptx {
// ---- the unit-independence precondition -------------------------------------
namespace {
bool In(std::string_view value, std::initializer_list<std::string_view> set) noexcept {
  for (const auto item : set)
    if (value == item) return true;
  return false;
}
// Helpers of the per-site scan rules (state space, semantics and scope).
std::string_view SpaceOf(const std::vector<std::string> &mods) noexcept {
  for (const auto &p : mods) {
    if (p == "global") return "global";
    if (p.starts_with("shared")) return "shared";
    if (p == "local" || p == "const" || p.starts_with("param")) return "other";
  }
  return "generic";
}
struct SemScope {
  std::string_view sem, scope;
};
SemScope SemScopeOf(const std::vector<std::string> &mods) noexcept {
  SemScope result;
  for (const auto &p : mods) {
    if (result.sem.empty() &&
        In(p, {"relaxed", "acquire", "release", "acq_rel", "sc", "weak", "volatile", "mmio"}))
      result.sem = p;
    if (In(p, {"cta", "cluster", "gpu", "sys"})) result.scope = p;
  }
  return result;
}
bool Wide(std::string_view scope) noexcept { return scope == "gpu" || scope == "sys"; }
} // namespace

PreconditionReport PreconditionScan(const Function &function) {
  PreconditionReport report;
  const auto violation = [&] { report.holds = false; };
  for (std::size_t index = 0; index < function.instructions.size(); ++index) {
    const auto &i = function.instructions[index];
    const auto &op = i.opcode;
    const auto &mods = i.modifiers;
    if (op == "atom" || op == "red") {
      if (!mods.empty() && mods[0] == "async") continue; // red.async: shared::cluster only
      const auto space = SpaceOf(mods);
      if (space != "global" && space != "generic") continue;
      report.atomic_sites.push_back(static_cast<std::uint32_t>(index)); // global atomic: every scope, where executed
    } else if (op == "ld" || op == "st") {
      if (!mods.empty() && (mods[0] == "async" || mods[0] == "bulk")) continue;
      const auto space = SpaceOf(mods);
      if (space != "global" && space != "generic") continue;
      const auto [sem, scope] = SemScopeOf(mods);
      if (sem == "volatile" || sem == "mmio") violation(); // a system-scope strong access
      else if (In(sem, {"relaxed", "acquire", "release"}) && Wide(scope)) violation(); // strong, wider than the unit
    } else if (op == "multimem") {
      violation(); // a multimem access
    } else if (op == "fence") {
      const auto scope = SemScopeOf(mods).scope;
      if (!Wide(scope)) continue;
      // The proxy-fence exception: proxy fences only restrict base causality order; memory
      // fences (sc, acq_rel, acquire, release, mbarrier_init) stay rejected.
      if (mods.empty() || mods[0] != "proxy") violation(); // a memory fence wider than the unit
    } else if (op == "membar") {
      if (!mods.empty() && (mods[0] == "gl" || mods[0] == "sys")) violation(); // a memory fence wider than the unit
    } else if (op == "tensormap" && !mods.empty() && mods[0] == "cp_fenceproxy") {
      // A copy fused with fence.proxy.tensormap::generic.release.<scope>: an admitted proxy fence,
      // checked by the tensormap locality rule.
      ++report.tensormap_copies;
    } else if (op == "cp" && mods.size() >= 3 && mods[0] == "reduce" && mods[1] == "async" && mods[2] == "bulk") {
      for (std::size_t k = 3; k < mods.size(); ++k) {
        if (mods[k] == "global") {
          violation(); // a global bulk reduction
          break;
        }
        if (mods[k].starts_with("shared")) break;
      }
    }
  }
  return report;
}

bool TensormapInputOverlap(const std::vector<TensormapRegion> &regions,
                           const std::vector<UnitPlan::InputMap> &maps) noexcept {
  if (regions.empty()) return false;
  for (const auto &map : maps) {
    if (map.allocation.empty()) return true;
    for (const auto &region : regions)
      if (region.object == map.allocation &&
          (region.begin >= map.offset ? region.begin - map.offset : map.offset - region.begin) < 128)
        return true; // two 128-byte regions overlap (written without overflow)
  }
  return false;
}

// ---- the CTA-row shard allowlist ---------------------------------------------
namespace {
bool IntegerType(std::string_view t) noexcept {
  return In(t, {"s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64", "b8", "b16", "b32", "b64"});
}
bool BitType(std::string_view t) noexcept { return In(t, {"b8", "b16", "b32", "b64", "pred"}); }
bool IdentChar(char c) noexcept {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
std::string_view Strip(std::string_view s) noexcept {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
  return s;
}
// The registers a function declares (`.reg .b64 %rd<105>;` declares %rd0..%rd104; `.reg .b32 %r;`
// declares %r): each name with its count (0: the plain name only).
using Registers = std::map<std::string, std::uint64_t, std::less<>>;
Registers DeclaredRegisters(const Module &module, const Function &function) {
  Registers out;
  for (const auto &d : function.directives) {
    if (d.name != ".reg") continue;
    const auto text = module.view(d.arguments);
    for (std::size_t at = 0; at < text.size(); ++at) {
      if (text[at] != '%') continue;
      auto stop = at + 1;
      while (stop < text.size() && IdentChar(text[stop])) ++stop;
      std::string name(text.substr(at, stop - at));
      std::uint64_t count = 0;
      if (stop < text.size() && text[stop] == '<') {
        auto close = text.find('>', stop);
        if (close == std::string_view::npos) continue;
        count = std::stoull(std::string(text.substr(stop + 1, close - stop - 1)));
        stop = close;
      }
      out[name] = std::max(out[name], count);
      at = stop;
    }
  }
  return out;
}
bool IsRegister(std::string_view s, const Registers &regs) {
  if (s.size() < 2 || s[0] != '%') return false;
  if (const auto it = regs.find(s); it != regs.end() && it->second == 0) return true;
  auto digits = s.size();
  while (digits > 1 && std::isdigit(static_cast<unsigned char>(s[digits - 1]))) --digits;
  if (digits == s.size()) return false;
  const auto it = regs.find(s.substr(0, digits));
  if (it == regs.end() || it->second == 0) return false;
  return std::stoull(std::string(s.substr(digits))) < it->second;
}
bool IsImmediate(std::string_view s) noexcept {
  if (!s.empty() && s[0] == '-') s.remove_prefix(1);
  if (!s.empty() && (s.back() == 'U' || s.back() == 'u')) s.remove_suffix(1);
  if (s.empty()) return false;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    for (const auto c : s.substr(2))
      if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    return true;
  }
  for (const auto c : s)
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  return true;
}
// The special registers a shard may read: launch shape and CTA/thread coordinates only.
bool IsListedSpecial(std::string_view s) noexcept {
  if (s == "%laneid") return true;
  for (const std::string_view base : {"%tid", "%ntid", "%ctaid", "%nctaid"})
    if (s.size() == base.size() + 2 && s.starts_with(base) && s[base.size()] == '.' &&
        In(s.substr(base.size() + 1), {"x", "y", "z"}))
      return true;
  return false;
}
// `[reg]`, `[reg+imm]` or `[reg+-imm]`; for ld.param, `[param]` or `[param+imm]` of `params`.
bool IsAddress(std::string_view s, const Registers &regs, const std::set<std::string, std::less<>> *params) {
  if (s.size() < 3 || s.front() != '[' || s.back() != ']') return false;
  s = Strip(s.substr(1, s.size() - 2));
  auto base = s, offset = std::string_view();
  if (const auto plus = s.find('+'); plus != std::string_view::npos) {
    base = Strip(s.substr(0, plus));
    offset = Strip(s.substr(plus + 1));
    if (!IsImmediate(offset)) return false;
  }
  return params ? params->contains(base) : IsRegister(base, regs);
}
// `{r, r, ...}` of registers.
bool IsVector(std::string_view s, const Registers &regs) {
  if (s.size() < 3 || s.front() != '{' || s.back() != '}') return false;
  s = s.substr(1, s.size() - 2);
  for (std::size_t begin = 0; begin <= s.size();) {
    auto end = s.find(',', begin);
    if (end == std::string_view::npos) end = s.size();
    if (!IsRegister(Strip(s.substr(begin, end - begin)), regs)) return false;
    begin = end + 1;
  }
  return true;
}
} // namespace

ShardScanReport ShardScan(const Module &module, const Function &entry) {
  ShardScanReport report;
  std::vector<const Function *> work{&entry};
  std::set<std::string> seen{entry.name};
  const auto refuse = [&](const Function &f, const Instruction &i, std::string why) {
    report.admitted = false;
    report.function = f.name;
    report.line = i.line;
    report.reason = std::move(why);
  };
  for (std::size_t next = 0; next < work.size() && report.admitted; ++next) {
    const auto &f = *work[next];
    report.scanned.push_back(f.name);
    const auto regs = DeclaredRegisters(module, f);
    std::set<std::string, std::less<>> params;
    for (const auto &p : f.parameters)
      if (p.name) params.insert(*p.name);
    for (const auto &p : f.returns)
      if (p.name) params.insert(*p.name);
    for (const auto &i : f.instructions) {
      const auto &op = i.opcode;
      const auto &m = i.modifiers;
      std::vector<std::string_view> operands;
      for (const auto &r : i.operands) operands.push_back(Strip(module.view(r)));
      const auto spelled = [&] {
        std::string s = op;
        for (const auto &x : m) s += "." + x;
        return s;
      };
      if (i.predicate && !IsRegister(*i.predicate, regs)) {
        refuse(f, i, "guard " + *i.predicate + " is not a declared register");
        break;
      }
      const auto value = [&](std::string_view s) { return IsRegister(s, regs) || IsImmediate(s); };
      const auto all_values = [&](std::size_t from) {
        for (std::size_t k = from; k < operands.size(); ++k)
          if (!value(operands[k])) return false;
        return true;
      };
      const auto dest = [&] { return !operands.empty() && IsRegister(operands[0], regs); };
      bool ok = false;
      if (In(op, {"add", "sub", "div", "rem", "min", "max", "neg", "abs", "selp", "shl", "shr"}))
        ok = m.size() == 1 && IntegerType(m[0]) && dest() && all_values(1);
      else if (In(op, {"mul", "mad"}))
        ok = m.size() == 2 && In(m[0], {"lo", "hi", "wide"}) && IntegerType(m[1]) && dest() && all_values(1);
      else if (In(op, {"and", "or", "xor", "not"}))
        ok = m.size() == 1 && BitType(m[0]) && dest() && all_values(1);
      else if (op == "setp")
        ok = m.size() == 2 && In(m[0], {"eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs"}) &&
             IntegerType(m[1]) && operands.size() == 3 && dest() && all_values(1);
      else if (op == "cvt")
        ok = m.size() == 2 && IntegerType(m[0]) && IntegerType(m[1]) && operands.size() == 2 && dest() &&
             all_values(1);
      else if (op == "mov")
        ok = m.size() == 1 && (IntegerType(m[0]) || m[0] == "pred") && operands.size() == 2 && dest() &&
             (value(operands[1]) || IsListedSpecial(operands[1]));
      else if (op == "bra")
        ok = (m.empty() || (m.size() == 1 && m[0] == "uni")) && i.control == "direct_branch" &&
             i.branch_target.has_value() && i.indirect_targets.empty();
      else if (op == "ret" || op == "exit")
        ok = m.empty() && operands.empty();
      else if (op == "call") {
        // Direct call only: a callee name of a function with a body; its scan joins this one.
        // Argument and return staging (st.param) is not listed, so a call passing values refuses.
        const Function *callee = nullptr;
        bool indirect = false;
        for (const auto o : operands) {
          if (o.empty() || o.front() == '(') continue;
          if (o.front() == '%') {
            indirect = true;
            continue;
          }
          for (const auto &g : module.functions)
            if (g.name == o && g.has_body) callee = &g;
        }
        ok = (m.empty() || (m.size() == 1 && m[0] == "uni")) && !indirect && callee;
        if (ok && seen.insert(callee->name).second) work.push_back(callee);
      } else if (op == "ld") {
        if (m.size() >= 2 && m[0] == "param")
          ok = (m.size() == 2 || (m.size() == 3 && In(m[1], {"v2", "v4"}))) && IntegerType(m.back()) &&
               operands.size() == 2 && (dest() || IsVector(operands[0], regs)) && IsAddress(operands[1], regs, &params);
        else if (m.size() >= 2 && m[0] == "global")
          ok = (m.size() == 2 || (m.size() == 3 && In(m[1], {"v2", "v4", "v8"}))) && IntegerType(m.back()) &&
               operands.size() == 2 && (dest() || IsVector(operands[0], regs)) && IsAddress(operands[1], regs, nullptr);
      } else if (op == "st")
        ok = m.size() >= 2 && m[0] == "global" &&
             (m.size() == 2 || (m.size() == 3 && In(m[1], {"v2", "v4", "v8"}))) && IntegerType(m.back()) &&
             operands.size() == 2 && IsAddress(operands[0], regs, nullptr) &&
             (value(operands[1]) || IsVector(operands[1], regs));
      else if (op == "cvta")
        ok = m.size() == 3 && m[0] == "to" && m[1] == "global" && In(m[2], {"u32", "u64"}) &&
             operands.size() == 2 && dest() && IsRegister(operands[1], regs);
      if (!ok) {
        std::string text = spelled();
        for (std::size_t k = 0; k < operands.size(); ++k) text += (k ? ", " : " ") + std::string(operands[k]);
        refuse(f, i, "not on the shard allowlist: " + text);
        break;
      }
    }
  }
  return report;
}

// ---- units and the plan -----------------------------------------------------
UnitPlan PlanUnits(const FrontendConfig &config, const Function &function,
                   const UnitGates &gates) {
  UnitPlan plan;
  const auto grid = LaunchDimensions(config).first;
  plan.grid = grid;
  plan.unit_shape = config.cluster_dim.value_or(Coordinates{1, 1, 1});
  plan.precondition = PreconditionScan(function);
  // An allocation_location names a launch resource; the slot resolver binds it to the resource's
  // concrete allocation object at its base offset (slot_resolver.cpp MemoryResource), or to the
  // resource id itself for a pointer without one. An unknown resource cannot be placed (empty name).
  using Json = nlohmann::json;
  const auto bound = [&](const std::string &id, std::uint64_t offset) -> UnitPlan::InputMap {
    for (const auto &r : config.resources) {
      if (!r.is_object() || r.value("id", std::string()) != id) continue;
      const auto allocation = r.value("allocation", Json::object());
      if (allocation.empty()) return {id, offset};
      if (allocation.value("resolution", "") != "concrete" || !allocation.contains("object") ||
          !allocation.contains("base_byte_offset"))
        return {};
      const auto base = JsonInteger(allocation["base_byte_offset"]);
      if (base < 0 || base > UINT64_MAX - offset) return {};
      return {allocation["object"].get<std::string>(), base.convert_to<std::uint64_t>() + offset};
    }
    return {};
  };
  for (auto it = config.tma_descriptors.begin(); it != config.tma_descriptors.end(); ++it) {
    const auto &d = it.value();
    if (!d.is_object()) continue;
    if (const auto location = d.find("allocation_location");
        location != d.end() && location->is_array() && location->size() == 2 && (*location)[0].is_string()) {
      const auto offset = JsonInteger((*location)[1]);
      if (offset < 0 || offset > UINT64_MAX) plan.input_maps.push_back({});
      else plan.input_maps.push_back(bound((*location)[0].get<std::string>(), offset.convert_to<std::uint64_t>()));
    } else if (const auto address = d.find("address"); address != d.end()) {
      const auto value = JsonInteger(*address);
      plan.input_maps.push_back({{}, value >= 0 && value <= UINT64_MAX ? value.convert_to<std::uint64_t>() : 0});
    }
  }
  bool exact = true;
  Coordinates unit_grid{1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    if (plan.unit_shape[axis] == 0 || grid[axis] % plan.unit_shape[axis]) exact = false;
    else unit_grid[axis] = grid[axis] / plan.unit_shape[axis];
  }
  const std::uint64_t units = exact ? std::uint64_t(unit_grid[0]) * unit_grid[1] * unit_grid[2] : 0;
  // The units of every multi-unit launch with whole-CTA analysis, including
  // the ones class mode keeps off.
  plan.active = exact && units > 1 && !gates.value_thread_range;
  // Class mode needs a cluster grid that tiles the launch, more than one unit, none of the run
  // options in UnitGates and the unit-independence precondition. Values are lifted through the
  // scalar fold; without it nothing lifts them.
  plan.class_mode = plan.active && !gates.trusted_reference && !gates.value_dump &&
                    !gates.branch_scope && !gates.scalar_fold_dump && !gates.exhaustive_search &&
                    !gates.continue_search && !gates.development_exports && !gates.cta_rows && plan.precondition.holds &&
                    !(gates.value_analysis && !gates.scalar_fold);
  // Class mode with value analysis implies the scalar fold.
  plan.lift_values = plan.class_mode && gates.value_analysis;
  if (!plan.active) return plan;
  plan.units.reserve(units);
  const auto &d = plan.unit_shape;
  for (std::uint32_t z = 0; z < unit_grid[2]; ++z)
    for (std::uint32_t y = 0; y < unit_grid[1]; ++y)
      for (std::uint32_t x = 0; x < unit_grid[0]; ++x) {
        LaunchUnit unit;
        unit.index = static_cast<std::uint32_t>(plan.units.size());
        unit.kappa = {x, y, z};
        unit.ctas.reserve(plan.UnitSize());
        for (std::uint32_t lz = 0; lz < d[2]; ++lz)
          for (std::uint32_t ly = 0; ly < d[1]; ++ly)
            for (std::uint32_t lx = 0; lx < d[0]; ++lx)
              unit.ctas.push_back({x * d[0] + lx, y * d[1] + ly, z * d[2] + lz});
        plan.units.push_back(std::move(unit));
      }
  return plan;
}

// ---- the structure key ------------------------------------------------------
namespace {
constexpr std::uint64_t Mix(std::uint64_t z) noexcept {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
template <class T> struct IsOptional : std::false_type {};
template <class T> struct IsOptional<std::optional<T>> : std::true_type {};
// Appends a canonical byte encoding (fixed-width integers, length-prefixed
// strings and sequences). Two keys are equal iff their bytes are equal.
class KeyWriter {
public:
  explicit KeyWriter(std::string &bytes) noexcept : bytes_(bytes) {}
  void U64(std::uint64_t value) {
    char raw[8];
    std::memcpy(raw, &value, 8);
    bytes_.append(raw, 8);
  }
  void Str(std::string_view text) {
    U64(text.size());
    bytes_.append(text);
  }
  template <class T> void Put(const T &x) {
    if constexpr (std::is_same_v<T, bool>) U64(x ? 1 : 2);
    else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) U64(static_cast<std::uint64_t>(x));
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) Str(x);
    else if constexpr (std::is_same_v<T, Integer>) Str(x.str());
    else if constexpr (std::is_same_v<T, DynamicKey>) {
      U64(x.instruction);
      U64(x.occurrence);
    } else if constexpr (std::is_same_v<T, BranchScopeDecision>) {
      Put(x.taken);
      U64(x.visit);
      U64(x.user_decision);
    } else if constexpr (std::is_same_v<T, PathId>) U64(x.value);
    else if constexpr (std::is_same_v<T, TemplatePathNode>) {
      Put(x.parent);
      U64(x.prefix);
      U64(x.pc);
      Put(x.reason);
      Put(x.group);
    } else if constexpr (std::is_same_v<T, DataBranchDecision>) {
      Str(x.assumption);
      Str(x.rule);
      U64(x.taken);
      U64(x.not_taken);
    } else if constexpr (IsOptional<T>::value) {
      if (x) {
        U64(1);
        Put(*x);
      } else U64(0);
    } else if constexpr (requires { x.first; x.second; }) {
      Put(x.first);
      Put(x.second);
    } else {
      U64(x.size());
      for (const auto &element : x) Put(element);
    }
  }
  // Every ExecutionMetadata field of one record.
  void Record(const RecordView &r) {
    Put(r.instruction_id()); Put(r.occurrence()); Put(r.predicate_value()); Put(r.branch_scope_decision());
    Put(r.collapsed_wait_loop()); Put(r.normalized_wait_result()); Put(r.normalized_clc_result());
    Put(r.canonical_wait_probe()); Put(r.wait_exit_label()); Put(r.schedule_seed_outcome());
    Put(r.probe_fallback_key()); Put(r.opaque_compute_diamond_indices()); Put(r.collapsed_wait_indices());
    Put(r.opaque_compute_join_label()); Put(r.assumed_equality()); Put(r.opaque_compute_register_reads());
    Put(r.opaque_compute_register_writes()); Put(r.register_origins()); Put(r.merged_arm_join());
  }

private:
  std::string &bytes_;
};
} // namespace

StructureKey ComputeStructureKey(const CtaFacts &facts) {
  StructureKey key;
  KeyWriter w(key.bytes);
  const bool templates = facts.TemplateExecution();
  w.Put(templates);
  w.U64(facts.LaneCount());
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto lane = facts.Lane(index);
    w.U64(facts.Warp(lane));
    w.U64(facts.LaneId(lane));
    w.Put(facts.TemplatePath(lane));
    w.U64(facts.Size(lane));
  }
  const auto tree = facts.TemplatePaths();
  w.Put(tree);
  w.Put(facts.GroupPaths());
  const auto groups = facts.TemplateGroups();
  w.Put(groups);
  // One metadata sequence per path (a member of its finished group), or per lane on the lane path.
  std::vector<RegId> bound;
  const auto sequence = [&](LaneRef lane, bool prototype) {
    const auto size = facts.Size(lane);
    w.U64(size);
    for (std::size_t pos = 0; pos < size; ++pos) {
      w.Record(facts.Meta({lane, pos}));
      if (prototype) {
        facts.PrototypeRegisterIds({lane, pos}, bound);
        w.Put(bound);
      }
    }
  };
  if (templates) {
    for (const auto &node : tree)
      if (node.group && *node.group < groups.size() && !groups[*node.group].empty()) {
        w.U64(1);
        sequence(facts.Lane(groups[*node.group].front()), true);
      } else w.U64(0);
  } else
    for (std::size_t index = 0; index < facts.LaneCount(); ++index) sequence(facts.Lane(index), false);
  for (std::size_t index = 0; index < facts.LaneCount(); ++index) {
    const auto aggregates = facts.Aggregates(facts.Lane(index));
    w.U64(aggregates.assumed_assertions);
    w.Put(aggregates.opaque_regions);
    w.Put(aggregates.data_branch_decisions);
  }
  w.Put(facts.StaticCounts());
  const auto sites = facts.SharedSitesRelative();
  w.U64(sites.size());
  for (const auto &site : sites) {
    w.Put(site.owner ? std::optional<std::uint32_t>(site.owner->value) : std::nullopt);
    w.U64(site.thread);
    w.Put(site.instruction);
  }
  std::uint64_t h = 0x243f6a8885a308d3ULL;
  std::size_t i = 0;
  for (; i + 8 <= key.bytes.size(); i += 8) {
    std::uint64_t word;
    std::memcpy(&word, key.bytes.data() + i, 8);
    h = Mix(h ^ Mix(word));
  }
  std::uint64_t tail = 0;
  if (i < key.bytes.size()) std::memcpy(&tail, key.bytes.data() + i, key.bytes.size() - i);
  key.hash = Mix(h ^ Mix(tail ^ key.bytes.size()));
  return key;
}
} // namespace sentryward::frontend::ptx
