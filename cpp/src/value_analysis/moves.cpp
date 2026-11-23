// Moves mode: see moves.h and docs/moves-mode.md.
#include "sentryward/value_analysis/moves.h"

#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/type_conversion.h"

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
using Json = nlohmann::json;

struct Stop final : std::runtime_error {
  std::string result, code;
  Json detail;
  Stop(std::string r, std::string c, const std::string& message, Json d)
      : std::runtime_error(message), result(std::move(r)), code(std::move(c)), detail(std::move(d)) {}
};
[[noreturn]] void refuse(const std::string& code, const std::string& message, Json d = Json::object()) {
  throw Stop("refused", code, message, std::move(d));
}
[[noreturn]] void unsupported(const std::string& code, const std::string& message, Json d = Json::object()) {
  throw Stop("unsupported", code, message, std::move(d));
}

// ---- The term (schema sentryward.moves-term/v1). ----
struct Units {
  std::string object, allocation;
  std::uint64_t base = 0, stride = 0, units = 0;
};
struct Index {
  std::string object, element;
  std::uint64_t offset = 0, stride = 0, element_bytes = 0;
  std::int64_t none = -1;
};
struct Term {
  std::string row;
  std::uint64_t unit_bytes = 0;
  Units source, destination;
  Index index;
  bool live = true;  // "all": every source unit's index is live; "none": every one is the excluded value
};

std::uint64_t elementBytes(const std::string& e) {
  if (e == "s8" || e == "u8") return 1;
  if (e == "s16" || e == "u16") return 2;
  if (e == "s32" || e == "u32") return 4;
  if (e == "s64" || e == "u64") return 8;
  throw std::invalid_argument("term: index element " + e + " is not an integer type");
}

Units parseUnits(const Json& j) {
  Units u;
  u.object = j.at("object");
  u.allocation = AllocationIdentity("global", u.object);
  u.base = j.at("base_bytes");
  u.stride = j.at("unit_stride_bytes");
  u.units = j.at("units");
  return u;
}

Term parseTerm(const Json& j) {
  if (j.at("schema") != "sentryward.moves-term/v1") throw std::invalid_argument("term: schema");
  Term t;
  t.row = j.value("row", "");
  t.unit_bytes = j.at("unit_bytes");
  t.source = parseUnits(j.at("source"));
  t.destination = parseUnits(j.at("destination"));
  const auto& i = j.at("destination").at("index");
  t.index.object = i.at("object");
  t.index.element = i.at("element");
  t.index.element_bytes = elementBytes(t.index.element);
  t.index.offset = i.at("offset_bytes");
  t.index.stride = i.at("stride_bytes");
  t.index.none = i.at("none");
  const auto& live = j.at("live");
  if (live == "all") t.live = true;
  else if (live == "none") t.live = false;
  else throw std::invalid_argument("term: live is \"all\" or \"none\"");
  if (!t.unit_bytes || t.unit_bytes > t.source.stride || t.unit_bytes > t.destination.stride)
    throw std::invalid_argument("term: a unit is wider than its stride");
  if (!t.index.stride || t.index.stride % t.index.element_bytes || t.index.offset % t.index.element_bytes)
    throw std::invalid_argument("term: index layout");
  if (t.source.object == t.destination.object) throw std::invalid_argument("term: source and destination are one object");
  return t;
}

// ---- The check. ----
class Checker {
 public:
  Checker(const v::Graph& g, const MemoryEliminationResult& m, const Term& t)
      : g_(g), m_(m), t_(t), spec_(g, false) {}

  // The bytes a value holds, as initial contents of one allocation: [offset, offset + bytes).
  struct Source {
    std::string allocation;
    std::uint64_t offset = 0, bytes = 0;
  };

  v::Value resolve(v::Value x) { return spec_.selected(g_.canonicalInterpretation(x)); }

  std::string describe(v::Value x) {
    x = resolve(x);
    if (x.conversion())
      return std::string(x.conversion()->spec->kind == v::TypeConversionKind::Convert ? "cvt." : "bitcast.") +
             x.conversion()->spec->target.toString();
    return std::string(g_.definingNode(x).operation().name()) + ":" + g_.type(x.withoutSelection()).toString();
  }

  // The initial-content bytes that value x holds bit for bit: x is a load (or a constant component
  // of one) resolved to initial contents, or to a stored value that is itself such a value, through
  // same-width bitcasts only. Anything else (a conversion, an operation, a partial or reassembled
  // read, a read at a symbolic position) is refused: its bits are not a copy.
  Source source(v::Value x, const Json& where, unsigned depth = 0) {
    if (depth > 64) unsupported("depth", "a copy chain deeper than 64 steps", where);
    x = resolve(x);
    if (x.threadSelector()) unsupported("thread_selector", "a thread selector the launch does not decide", where);
    const auto bits = g_.type(x).bitSize();
    if (const auto& c = x.conversion()) {
      if (c->spec->kind != v::TypeConversionKind::Bitcast || g_.type(c->source).bitSize() != bits) {
        auto d = where;
        d["found"] = describe(x);
        refuse("value", "the written value is converted, not copied", d);
      }
      return source(c->source, where, depth + 1);
    }
    const auto& op = g_.definingNode(x).operation();
    if (!dynamic_cast<const v::LoadOp*>(&op)) {
      auto d = where;
      d["found"] = describe(x);
      refuse("value", "the written value is computed, not loaded", d);
    }
    if (bits % 8) unsupported("value_width", "a loaded value that is not whole bytes", where);
    const MemoryInstance key{x.nodeId(), x.instanceIndex()};
    std::uint64_t sel = 0;
    if (!x.indices().empty()) {
      const auto k = x.indices().size() == 1 ? spec_.known(x.indices()[0]) : std::nullopt;
      if (!k) unsupported("load_component", "a selected component of a load is not a constant", where);
      sel = std::stoull(k->literal) * (bits / 8);
    }
    const auto region = [&](const ConstantRegion& r) {
      return Source{r.allocation, r.offset + sel, bits / 8};
    };
    if (auto it = m_.symbolic_initial_reads.find(key); it != m_.symbolic_initial_reads.end())
      return region(it->second);
    if (m_.symbolic_read_positions.contains(key))
      unsupported("symbolic_read_position", "a read at a data-dependent position of a written object", where);
    if (const auto* r = m_.readReplacement(key)) {
      if (const auto* stored = std::get_if<v::Value>(r)) {
        auto s = *stored;
        if (!x.indices().empty()) s = g_.component(s, {x.indices().begin(), x.indices().end()});
        return source(s, where, depth + 1);
      }
      if (const auto* initial = std::get_if<InitialMemorySymbol>(r)) return region(initial->region);
      refuse("partial_read", "a reinterpreted or reassembled read (not one whole loaded value)", where);
    }
    if (const auto initial = m_.initialRead(key)) return region(initial->region);
    unsupported("unresolved_load", "a load with no read resolution", where);
  }

  Json run() {
    const auto& src = t_.source;
    const auto& dst = t_.destination;
    const auto& ix = t_.index;
    std::uint64_t writes = 0, moved_bytes = 0;
    std::map<std::string, std::uint64_t> other;  // writes to other objects, by allocation
    // Bytes written per (source unit, byte in unit): a byte written twice is refused.
    std::vector<std::vector<bool>> written(src.units, std::vector<bool>(t_.unit_bytes, false));
    m_.forEachOutput(g_, [&](const OutputWrite& w) {
      if (w.region.allocation != dst.allocation) {
        ++other[w.region.allocation];
        return;
      }
      ++writes;
      Json where = {{"offset", w.region.offset}, {"bytes", w.region.bytes}};
      if (w.reduced) unsupported("output_form", "a reduced destination cell", where);
      // The excluded-value guard: with every index at the excluded value no unit is live.
      if (!t_.live) refuse("guard", "a write to the destination while every index is the excluded value", where);
      // The position: one symbolic term, the index table's element for some source unit r, times
      // the destination's unit stride.
      if (w.base.size() != 1) {
        where["terms"] = w.base.size();
        refuse("index_map", "a destination write not placed by one index-table element", where);
      }
      const auto& term = w.base[0];
      const auto element = static_cast<__int128>(term.element_offset);  // bytes (SymbolicAddressTerm)
      where["index_object"] = term.object;
      where["index_byte"] = static_cast<std::int64_t>(element);
      where["coefficient"] = term.coefficient;
      if (term.object != ix.object || term.element_type != ix.element || element < ix.offset ||
          (element - ix.offset) % ix.stride || static_cast<__int128>(term.coefficient) != dst.stride)
        refuse("index_map", "a destination write not placed by the index map", where);
      const auto r = static_cast<std::uint64_t>((element - ix.offset) / ix.stride);
      where["unit"] = r;
      if (r >= src.units) refuse("index_map", "an index element past the source units", where);
      if (w.region.offset < dst.base || w.region.offset - dst.base + w.region.bytes > t_.unit_bytes)
        refuse("unit_bounds", "a write outside its destination unit", where);
      const auto u = w.region.offset - dst.base;
      // The value: exactly bytes [u, u + bytes) of source unit r.
      const auto s = source(w.value, where);
      where["source"] = {{"allocation", s.allocation}, {"offset", s.offset}, {"bytes", s.bytes}};
      if (w.value_byte_offset + w.region.bytes > s.bytes)
        unsupported("output_granularity", "a write of bytes past its value", where);
      const auto from = s.offset + w.value_byte_offset;
      if (s.allocation != src.allocation || from != src.base + r * src.stride + u)
        refuse("value", "the written bytes are not the same bytes of the source unit", where);
      for (std::uint64_t b = u; b < u + w.region.bytes; ++b) {
        if (written[r][b]) refuse("duplicate_write", "a destination byte written twice", where);
        written[r][b] = true;
      }
      moved_bytes += w.region.bytes;
    });
    Json others = Json::object();
    for (const auto& [a, n] : other) others[a] = n;
    return {{"result", "matched"},
            {"moves", {{"live", t_.live ? "all" : "none"}, {"units", src.units}, {"unit_bytes", t_.unit_bytes},
                       {"writes", writes}, {"moved_bytes", moved_bytes}, {"other_writes", others}}}};
  }

 private:
  const v::Graph& g_;
  const MemoryEliminationResult& m_;
  const Term& t_;
  ScalarSpecialization spec_;
};

}  // namespace

std::string checkMovesTerm(const value_ir::Graph& graph, const MemoryEliminationResult& memory,
                           const std::string& term_json, const UnitLifting* lifting) {
  Json report{{"schema", "sentryward.moves-report/v1"}};
  try {
    const auto term = parseTerm(Json::parse(term_json));
    report["row"] = term.row;
    if (!memory.complete)
      unsupported("memory_elimination", "memory elimination is not complete",
                  memory.diagnostic ? Json{{"code", memory.diagnostic->code}, {"message", memory.diagnostic->message}}
                                    : Json::object());
    if (lifting) unsupported("class_mode", "class mode (CTA class representatives only) is not supported");
    Checker checker(graph, memory, term);
    report.update(checker.run());
  } catch (const Stop& s) {
    report["result"] = s.result;
    report["reason"] = s.code;
    report["message"] = s.what();
    report["detail"] = s.detail;
  } catch (const std::exception& e) {
    report["result"] = "unsupported";
    report["reason"] = "term_or_internal";
    report["message"] = e.what();
  }
  return report.dump(1);
}

}  // namespace sentryward::value_analysis
