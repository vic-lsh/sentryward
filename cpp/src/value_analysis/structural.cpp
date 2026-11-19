// Structural mode: see structural.h and docs/structural-mode.md.
#include "sentryward/value_analysis/structural.h"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "sentryward/value_analysis/scalar_specialization.h"
#include "sentryward/value_ir/operations.h"
#include "sentryward/value_ir/type_conversion.h"

namespace sentryward::value_analysis {
namespace {
namespace v = value_ir;
using Json = nlohmann::json;

// A stop of the check: a refusal (the launch does not realize the term) or an unsupported case
// (the check cannot decide). Neither is ever a match.
struct Stop final : std::runtime_error {
  std::string result, code;
  Json detail;
  Stop(std::string r, std::string c, const std::string& message, Json d = Json::object())
      : std::runtime_error(message), result(std::move(r)), code(std::move(c)), detail(std::move(d)) {}
};
[[noreturn]] void refuse(const std::string& code, const std::string& message, Json d = Json::object()) {
  throw Stop("refused", code, message, std::move(d));
}
[[noreturn]] void unsupported(const std::string& code, const std::string& message, Json d = Json::object()) {
  throw Stop("unsupported", code, message, std::move(d));
}

// ---- Leaf sets: sorted disjoint half-open intervals over leaf ids (space << 40 | element). ----
constexpr unsigned kShift = 40;
constexpr std::uint64_t kElementMask = (std::uint64_t{1} << kShift) - 1;
struct Interval {
  std::uint64_t begin, end;
  bool operator==(const Interval&) const = default;
};
using Set = std::vector<Interval>;
using SetRef = std::shared_ptr<const Set>;

Set normalize(std::vector<Interval> xs) {
  std::sort(xs.begin(), xs.end(), [](const Interval& a, const Interval& b) { return a.begin < b.begin; });
  Set out;
  for (const auto& x : xs) {
    if (x.begin >= x.end) continue;
    if (!out.empty() && x.begin <= out.back().end) out.back().end = std::max(out.back().end, x.end);
    else out.push_back(x);
  }
  return out;
}
Set unite(const Set& a, const Set& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  std::vector<Interval> all;
  all.reserve(a.size() + b.size());
  std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(all),
             [](const Interval& x, const Interval& y) { return x.begin < y.begin; });
  Set out;
  for (const auto& x : all) {
    if (!out.empty() && x.begin <= out.back().end) out.back().end = std::max(out.back().end, x.end);
    else out.push_back(x);
  }
  return out;
}

// ---- The term (schema sentryward.structural-term/v1). ----
struct Object {
  std::string name, element, allocation;
  std::uint64_t bytes = 0, stride = 0, rows = 0, cols = 0;
  bool row_indexed = true;  // rows follow the launch's row space (else a fixed table, e.g. a weight)
};
struct Index {
  enum Kind { Out, All, Const } kind;
  std::uint64_t value;
};
using Pattern = std::array<Index, 2>;  // (row, column)
struct Arg {
  bool body = false;
  std::size_t ref = 0;  // term input (an object) or an earlier term node
  std::string convert;  // the element type a primitive operand is converted to; empty: none
  std::vector<Pattern> reads;
};
struct TermNode {
  std::string op;
  bool primitive = false;
  std::string operation, type;
  bool commutative = false;
  std::vector<Arg> args;
  std::optional<std::size_t> out;  // the object holding this node's value
};
struct Term {
  std::string row;
  std::vector<Object> objects;
  std::vector<std::size_t> inputs;
  std::vector<TermNode> nodes;
};

std::size_t objectIndex(const Term& t, const std::string& name) {
  for (std::size_t i = 0; i < t.objects.size(); ++i)
    if (t.objects[i].name == name) return i;
  throw std::invalid_argument("term: unknown object " + name);
}

Term parseTerm(const Json& j) {
  if (j.at("schema") != "sentryward.structural-term/v1") throw std::invalid_argument("term: schema");
  Term t;
  t.row = j.value("row", "");
  for (const auto& o : j.at("objects")) {
    Object x;
    x.name = o.at("name");
    x.element = o.at("element");
    x.bytes = o.at("element_bytes");
    x.stride = o.at("row_stride_bytes");
    x.rows = o.at("rows");
    x.cols = o.at("cols");
    x.row_indexed = o.at("row_indexed");
    x.allocation = AllocationIdentity("global", x.name);
    if (!x.bytes || x.stride % x.bytes || x.cols * x.bytes > x.stride)
      throw std::invalid_argument("term: object layout " + x.name);
    t.objects.push_back(std::move(x));
  }
  for (const auto& i : j.at("inputs")) t.inputs.push_back(objectIndex(t, i));
  for (const auto& n : j.at("nodes")) {
    TermNode x;
    x.op = n.at("op");
    const auto& atom = n.at("atom");
    x.primitive = atom.at("kind") == "primitive";
    if (x.primitive) {
      x.operation = atom.at("operation");
      x.type = atom.at("type");
      x.commutative = atom.value("commutative", false);
    } else if (atom.at("kind") != "opaque") {
      throw std::invalid_argument("term: atom kind");
    }
    for (const auto& a : n.at("args")) {
      Arg y;
      if (a.contains("body")) {
        y.body = true;
        y.ref = a.at("body");
        if (y.ref >= t.nodes.size()) throw std::invalid_argument("term: body refers forward");
      } else {
        y.ref = a.at("input");
        if (y.ref >= t.inputs.size()) throw std::invalid_argument("term: input index");
      }
      if (a.contains("convert") && !a.at("convert").is_null()) y.convert = a.at("convert");
      for (const auto& p : a.at("reads")) {
        if (p.size() != 2) throw std::invalid_argument("term: a read pattern has two coordinates");
        Pattern q{};
        for (std::size_t d = 0; d < 2; ++d) {
          const auto& c = p.at(d);
          if (c.contains("out")) q[d] = {Index::Out, c.at("out")};
          else if (c.contains("all")) q[d] = {Index::All, c.at("all")};
          else if (c.contains("const")) q[d] = {Index::Const, c.at("const")};
          else throw std::invalid_argument("term: coordinate");
          if (q[d].kind == Index::Out && q[d].value > 1) throw std::invalid_argument("term: output coordinate");
        }
        y.reads.push_back(q);
      }
      x.args.push_back(std::move(y));
    }
    if (n.contains("out") && !n.at("out").is_null()) x.out = objectIndex(t, n.at("out"));
    t.nodes.push_back(std::move(x));
  }
  return t;
}

// ---- Expected leaves: product patterns over spaces, composed through body nodes. ----
struct SpacePattern {
  std::uint64_t space;  // an object index, or objects.size() + term node (that node's cut values)
  Pattern at;
};
Pattern compose(const Pattern& inner, const Pattern& outer) {
  Pattern r = inner;
  for (auto& c : r)
    if (c.kind == Index::Out) c = outer[c.value];
  return r;
}
// The patterns of node n's operands; with `cut`, a body operand is that node's cut values.
std::vector<SpacePattern> patterns(const Term& t, std::size_t n, bool cut) {
  std::vector<SpacePattern> out;
  for (const auto& a : t.nodes[n].args)
    for (const auto& p : a.reads) {
      if (!a.body) out.push_back({t.inputs[a.ref], p});
      else if (cut) out.push_back({t.objects.size() + a.ref, p});
      else
        for (const auto& q : patterns(t, a.ref, false)) out.push_back({q.space, compose(q.at, p)});
    }
  return out;
}

// ---- The check. ----
using Key = std::tuple<std::size_t, std::size_t, std::size_t>;  // node, result, instance

class Checker {
 public:
  Checker(const v::Graph& g, const MemoryEliminationResult& m, const Term& t, const UnitLifting* lifting)
      : g_(g), m_(m), t_(t), lifting_(lifting), spec_(g, lifting != nullptr) {}

  // The layout object of a space (a cut space has its node's output layout).
  const Object& layout(std::uint64_t space) const {
    if (space < t_.objects.size()) return t_.objects[space];
    return t_.objects.at(*t_.nodes.at(space - t_.objects.size()).out);
  }
  std::string spaceName(std::uint64_t space) const {
    if (space < t_.objects.size()) return t_.objects[space].name;
    if (space < t_.objects.size() + t_.nodes.size()) {
      const auto n = space - t_.objects.size();
      return "cut(" + t_.nodes[n].op + "#" + std::to_string(n) + ")";
    }
    return "allocation:" + foreign_.at(space - t_.objects.size() - t_.nodes.size());
  }
  // A leaf set as readable runs: space[row][begin..end).
  Json render(const Set& s, std::size_t limit = 16) const {
    Json out = Json::array();
    for (const auto& x : s) {
      if (out.size() >= limit) { out.push_back("..."); break; }
      const auto space = x.begin >> kShift;
      std::uint64_t b = x.begin & kElementMask, e = x.end - (space << kShift);
      std::ostringstream os;
      if (space < t_.objects.size() + t_.nodes.size()) {
        const auto& o = layout(space);
        const auto per = o.stride / o.bytes;
        os << spaceName(space) << "[" << b / per << "][" << b % per;
        if (e - b > 1) os << ".." << (b % per) + (e - b);
        os << "]";
      } else {
        os << spaceName(space) << " elements [" << b << ".." << e << ")";
      }
      out.push_back(os.str());
    }
    return out;
  }

  Set materialize(const std::vector<SpacePattern>& ps, std::uint64_t row, std::uint64_t col) const {
    std::vector<Interval> xs;
    const std::array<std::uint64_t, 2> at{row, col};
    for (const auto& p : ps) {
      const auto& o = layout(p.space);
      std::array<std::uint64_t, 2> lo{}, hi{};
      for (std::size_t d = 0; d < 2; ++d) {
        const auto& c = p.at[d];
        if (c.kind == Index::Out) lo[d] = at[c.value], hi[d] = lo[d] + 1;
        else if (c.kind == Index::Const) lo[d] = c.value, hi[d] = c.value + 1;
        else lo[d] = 0, hi[d] = c.value;
      }
      const auto per = o.stride / o.bytes;
      for (auto r = lo[0]; r < hi[0]; ++r)
        xs.push_back({(p.space << kShift) | (r * per + lo[1]), (p.space << kShift) | (r * per + hi[1])});
    }
    return normalize(std::move(xs));
  }

  // Leaves of the bytes [offset, offset + bytes) of allocation `allocation`'s initial contents.
  Set regionLeaves(const std::string& allocation, std::uint64_t offset, std::uint64_t bytes) {
    std::uint64_t space = 0, unit = 1;
    bool found = false;
    for (std::size_t i = 0; i < t_.objects.size(); ++i)
      if (t_.objects[i].allocation == allocation) space = i, unit = t_.objects[i].bytes, found = true;
    if (!found) {
      auto it = std::find(foreign_.begin(), foreign_.end(), allocation);
      space = t_.objects.size() + t_.nodes.size() + std::size_t(it - foreign_.begin());
      if (it == foreign_.end()) foreign_.push_back(allocation);
    }
    if (!bytes) return {};
    return {{(space << kShift) | (offset / unit), (space << kShift) | ((offset + bytes + unit - 1) / unit)}};
  }

  v::Value resolve(v::Value x) { return spec_.selected(g_.canonicalInterpretation(x)); }

  // The constant component offset (bytes) of a selection of a load's result, or none.
  std::optional<std::uint64_t> componentOffset(const v::Value& x, std::uint64_t element_bytes) {
    if (x.indices().empty()) return 0;
    if (x.indices().size() != 1) return {};
    const auto k = spec_.known(x.indices()[0]);
    if (!k) return {};
    try { return std::stoull(k->literal) * element_bytes; } catch (const std::exception&) { return {}; }
  }

  // Leaves of a load instance, through memory elimination's read resolution. `initial_only`:
  // a read resolved to a stored value is not a leaf (none is returned).
  std::optional<Set> loadLeaves(const v::Value& x, bool initial_only) {
    const MemoryInstance key{x.nodeId(), x.instanceIndex()};
    const auto& type = g_.type(x.withoutSelection());
    const auto element_bytes = std::uint64_t(type.element().width() / 8);
    const auto sel = componentOffset(x, element_bytes);
    if (!sel) unsupported("load_component", "a selected component of a load is not a constant");
    const auto narrow = [&](const std::string& alloc, std::uint64_t off, std::uint64_t bytes) {
      if (!x.indices().empty()) return regionLeaves(alloc, off + *sel, element_bytes);
      return regionLeaves(alloc, off, bytes);
    };
    if (auto it = m_.symbolic_initial_reads.find(key); it != m_.symbolic_initial_reads.end())
      return narrow(it->second.allocation, it->second.offset, it->second.bytes);
    if (m_.symbolic_read_positions.contains(key))
      unsupported("symbolic_read_position", "a read at a data-dependent position (needs table premises)");
    if (const auto* r = m_.readReplacement(key)) {
      if (const auto* value = std::get_if<v::Value>(r)) {
        if (initial_only) return std::nullopt;
        auto source = *value;
        if (!x.indices().empty()) source = g_.component(source, {x.indices().begin(), x.indices().end()});
        return leaves(source);
      }
      if (const auto* initial = std::get_if<InitialMemorySymbol>(r))
        return narrow(initial->region.allocation, initial->region.offset, initial->region.bytes);
      if (initial_only) return std::nullopt;
      if (!x.indices().empty()) unsupported("load_component", "a component of a reassembled read");
      const auto piece = [&](const ReinterpretedRead& p) -> Set {
        if (const auto* s = std::get_if<v::Value>(&p.source)) return leaves(*s);  // conservative: all of it
        const auto& i = std::get<InitialMemorySymbol>(p.source);
        return regionLeaves(i.region.allocation, i.region.offset + p.byte_offset, p.type.bitSize() / 8);
      };
      if (const auto* p = std::get_if<ReinterpretedRead>(r)) return piece(*p);
      Set s;
      for (const auto& p : std::get<AssembledRead>(*r).pieces) s = unite(s, piece(p));
      return s;
    }
    if (const auto initial = m_.initialRead(key))
      return narrow(initial->region.allocation, initial->region.offset, initial->region.bytes);
    unsupported("unresolved_load", "a load with no read resolution", {{"node", key.node}, {"instance", key.instance}});
  }

  // The leaf set of value x, cut values (stop_) counting as leaves. Memoized per plain value.
  Set leaves(v::Value x, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    if (x.threadSelector()) unsupported("thread_selector", "a thread selector the launch does not decide");
    if (x.conversion()) {
      auto s = leaves(x.conversion()->source, depth + 1);
      for (const auto& i : x.indices()) s = unite(s, leaves(i, depth + 1));
      return s;
    }
    const bool plain = x.indices().empty() && !x.pointer();
    const Key key{x.nodeId(), x.resultIndex(), x.instanceIndex()};
    if (plain) {
      if (auto it = stop_.find(key); it != stop_.end()) return {{it->second, it->second + 1}};
      if (auto it = memo_.find(key); it != memo_.end()) return *it->second;
    }
    const auto& node = g_.definingNode(x);
    const auto& op = node.operation();
    Set s;
    if (dynamic_cast<const v::LoadOp*>(&op)) {
      s = *loadLeaves(x, false);
    } else if (dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op) ||
               dynamic_cast<const v::CoordinateOp*>(&op)) {
    } else {
      const auto operands = g_.instanceOperands(x);
      for (const auto& a : operands) s = unite(s, leaves(a, depth + 1));
      for (const auto& i : x.indices()) s = unite(s, leaves(i, depth + 1));
    }
    if (plain) memo_.emplace(key, std::make_shared<const Set>(s));
    return s;
  }

  // A short readable form of x's top operations (for refusal details).
  std::string describe(v::Value x, unsigned depth = 3) {
    x = resolve(x);
    std::string out;
    if (x.conversion()) {
      out = std::string(x.conversion()->spec->kind == v::TypeConversionKind::Convert ? "cvt." : "bitcast.") +
            x.conversion()->spec->target.toString() + "(" + (depth ? describe(x.conversion()->source, depth - 1) : "..") + ")";
      return out;
    }
    const auto& node = g_.definingNode(x);
    out = std::string(node.operation().name()) + ":" + g_.type(x.withoutSelection()).toString();
    if (!x.indices().empty()) out += "[sel]";
    if (dynamic_cast<const v::LoadOp*>(&node.operation())) {
      const auto s = loadLeaves(x, false);
      return out + "{" + (s ? render(*s, 2).dump() : "stored") + "}";
    }
    if (!depth) return out;
    out += "(";
    const auto operands = g_.instanceOperands(x);
    for (std::size_t i = 0; i < operands.size(); ++i) out += (i ? ", " : "") + describe(operands[i], depth - 1);
    return out + ")";
  }

  // The single initial-content leaf x is, or none: x is (the conversion to `convert` of) a value of
  // the object's element type `element`, which is a load of that leaf or a same-width bitcast of one.
  std::optional<std::uint64_t> operandLeaf(v::Value x, const std::string& convert, const std::string& element) {
    x = resolve(x);
    if (!convert.empty()) {
      if (x.conversion() && x.conversion()->spec->kind == v::TypeConversionKind::Convert &&
          x.conversion()->spec->target.element().toString() == convert) {
        x = resolve(x.conversion()->source);
      } else if (!x.conversion() && dynamic_cast<const v::ConvertOp*>(&g_.definingNode(x).operation()) &&
                 g_.type(x).element().toString() == convert) {
        x = resolve(g_.instanceOperands(x).at(0));
      } else {
        return std::nullopt;
      }
    }
    if (g_.type(x).element().toString() != element) return std::nullopt;
    const auto width = g_.type(x).element().width();
    while (x.conversion() && x.conversion()->spec->kind == v::TypeConversionKind::Bitcast &&
           g_.type(x.conversion()->source).element().width() == width && x.indices().empty())
      x = resolve(x.conversion()->source);
    if (x.conversion() || x.threadSelector() || !dynamic_cast<const v::LoadOp*>(&g_.definingNode(x).operation()))
      return std::nullopt;
    const auto s = loadLeaves(x, true);
    if (!s || s->size() != 1 || s->front().end != s->front().begin + 1) return std::nullopt;
    return s->front().begin;
  }

  // Class mode: the representatives' written rows, and each member's rows as a representative's
  // shifted by k, where the member's offset of every row-indexed object is k of its rows and of
  // every other object zero. Together they must cover every row of every output object once.
  // The term's leaf patterns must then shift with the rows: a row-indexed object read at the
  // output's row only. Null without lifting.
  Json classMode(const std::map<std::size_t, std::set<std::uint64_t>>& written,
                 const std::map<std::uint64_t, std::set<std::uint64_t>>& by_cta) {
    if (!lifting_) return nullptr;
    std::optional<std::set<std::uint64_t>> rep_rows;
    std::uint64_t rows = 0;
    for (const auto& n : t_.nodes) {
      if (!n.out) continue;
      const auto it = written.find(*n.out);
      const auto here = it == written.end() ? std::set<std::uint64_t>{} : it->second;
      if (rep_rows && *rep_rows != here) refuse("class_rows", "output objects' representative rows differ");
      rep_rows = here;
      rows = t_.objects[*n.out].rows;
      for (std::size_t n2 = 0; n2 < t_.nodes.size(); ++n2)
        for (const bool cut : {false, true})
          for (const auto& p : patterns(t_, n2, cut))
            if (layout(p.space).row_indexed && p.at[0].kind != Index::Out)
              unsupported("class_mode_pattern", "a row-indexed object read at another row than the output's");
    }
    for (const auto& n : t_.nodes)
      if (n.out && (t_.objects[*n.out].rows != rows || !t_.objects[*n.out].row_indexed))
        unsupported("class_mode_rows", "output objects with different row spaces");
    const auto& L = *lifting_;
    std::vector<bool> covered(rows, false);
    std::size_t members = 0, member_rows = 0;
    const auto cover = [&](std::uint64_t r) {
      if (r >= rows || covered[r]) refuse("class_coverage", "a row covered twice or outside the launch", {{"row", r}});
      covered[r] = true;
    };
    std::map<UnitLifting::Origin, std::set<std::uint64_t>> by_origin;
    for (const auto& [cta, rs] : by_cta) {
      const UnitLifting::Origin c{cta % L.grid[0], cta / L.grid[0] % L.grid[1], cta / (L.grid[0] * L.grid[1])};
      UnitLifting::Origin o;
      for (std::size_t a = 0; a < 3; ++a) o[a] = c[a] / L.unit_shape[a] * L.unit_shape[a];
      by_origin[o].insert(rs.begin(), rs.end());
    }
    for (const auto& [origin, rs] : by_origin) {
      for (const auto r : rs) cover(r);
      const auto it = L.members.find(origin);
      if (it == L.members.end()) continue;
      for (const auto& m : it->second) {
        std::optional<std::int64_t> k;
        for (const auto& o : t_.objects) {
          const auto d = m.delta.contains(o.allocation) ? m.delta.at(o.allocation) : 0;
          if (!o.row_indexed) {
            if (d) refuse("class_shift", "a member moves a fixed object", {{"object", o.name}, {"delta", d}});
            continue;
          }
          if (d % std::int64_t(o.stride)) refuse("class_shift", "a member's offset is not whole rows", {{"object", o.name}, {"delta", d}});
          const auto ko = d / std::int64_t(o.stride);
          if (k && *k != ko) refuse("class_shift", "a member shifts objects by different rows", {{"object", o.name}, {"delta", d}});
          k = ko;
        }
        for (const auto& [alloc, d] : m.delta) {
          bool known = false;
          for (const auto& o : t_.objects) known |= o.allocation == alloc;
          if (!known && d) refuse("class_shift", "a member moves an allocation the term does not name", {{"allocation", alloc}});
        }
        if (!k) unsupported("class_shift", "no row-indexed object");
        ++members;
        for (const auto r : rs) {
          if (std::int64_t(r) + *k < 0) refuse("class_coverage", "a member row below zero");
          cover(std::uint64_t(std::int64_t(r) + *k));
          ++member_rows;
        }
      }
    }
    for (std::uint64_t r = 0; r < rows; ++r)
      if (!covered[r]) refuse("class_coverage", "a row no representative or member covers", {{"row", r}});
    return {{"representatives", by_origin.size()}, {"representative_rows", rep_rows ? rep_rows->size() : 0},
            {"members", members}, {"member_rows", member_rows}};
  }

  Json run() {
    // Stored values per output object element.
    std::map<std::size_t, std::vector<std::optional<v::Value>>> stored;
    for (const auto& n : t_.nodes)
      if (n.out) stored[*n.out].assign(t_.objects[*n.out].rows * (t_.objects[*n.out].stride / t_.objects[*n.out].bytes), std::nullopt);
    std::size_t writes = 0;
    std::map<std::size_t, std::set<std::uint64_t>> written_rows;
    std::map<std::uint64_t, std::set<std::uint64_t>> rows_by_cta;
    m_.forEachOutput(g_, [&](const OutputWrite& w) {
      ++writes;
      std::optional<std::size_t> obj;
      for (const auto& [o, _] : stored)
        if (t_.objects[o].allocation == w.region.allocation) obj = o;
      if (!obj) refuse("unexpected_output", "a write to an object no term node outputs", {{"allocation", w.region.allocation}, {"offset", w.region.offset}});
      const auto& o = t_.objects[*obj];
      if (w.reduced || !w.base.empty())
        unsupported("output_form", "a reduced or symbolically placed output", {{"object", o.name}});
      if (w.region.bytes != o.bytes || w.region.offset % o.bytes || w.value_byte_offset ||
          !g_.type(w.value).isScalar())
        unsupported("output_granularity", "an output write that is not one element of its object",
                    {{"object", o.name}, {"offset", w.region.offset}, {"bytes", w.region.bytes}});
      const auto e = w.region.offset / o.bytes;
      const auto per = o.stride / o.bytes;
      if (e / per >= o.rows || e % per >= o.cols)
        refuse("output_outside", "an output write outside the term's elements", {{"object", o.name}, {"offset", w.region.offset}});
      auto& slot = stored[*obj][e];
      if (slot) refuse("duplicate_output", "two final writes to one element", {{"object", o.name}, {"offset", w.region.offset}});
      slot = w.value;
      written_rows[*obj].insert(e / per);
      const auto cta = g_.node(w.writer.node).instanceOrigin(w.writer.instance).cta;
      if (lifting_ && !cta) unsupported("writer_cta", "an output write without a writer CTA");
      rows_by_cta[cta.value_or(0)].insert(e / per);
    });
    const auto class_mode = classMode(written_rows, rows_by_cta);
    Json nodes = Json::array();
    for (std::size_t n = 0; n < t_.nodes.size(); ++n) {
      const auto& tn = t_.nodes[n];
      if (!tn.out) {
        bool used = false;
        for (const auto& u : t_.nodes)
          for (const auto& a : u.args) used |= a.body && a.ref == n;
        if (used) unsupported("inner_without_output", "an inner term node without an output object (cut search not built)", {{"node", n}, {"op", tn.op}});
        continue;
      }
      const auto& o = t_.objects[*tn.out];
      const auto per = o.stride / o.bytes;
      const auto full = patterns(t_, n, false), cut = patterns(t_, n, true);
      const bool has_body = std::any_of(tn.args.begin(), tn.args.end(), [](const Arg& a) { return a.body; });
      // The cut leaves: each inner node's stored values, by element.
      std::map<Key, std::uint64_t> stop;
      if (has_body)
        for (const auto& a : tn.args) {
          if (!a.body) continue;
          const auto m = a.ref;
          const auto space = t_.objects.size() + m;
          const auto& values = stored.at(*t_.nodes[m].out);
          for (std::size_t e = 0; e < values.size(); ++e) {
            if (!values[e]) continue;
            auto x = resolve(*values[e]);
            if (x.conversion() || !x.indices().empty() || x.threadSelector())
              unsupported("cut_value_form", "an inner node's stored value is not a plain value", {{"node", m}});
            const Key k{x.nodeId(), x.resultIndex(), x.instanceIndex()};
            const auto id = (space << kShift) | e;
            if (auto [it, added] = stop.emplace(k, id); !added && it->second != id)
              refuse("ambiguous_cut", "one value stored at two elements of an inner node", {{"node", m}});
          }
        }
      Checker cutter(g_, m_, t_, lifting_);
      cutter.stop_ = std::move(stop);
      std::size_t checked = 0;
      for (std::uint64_t r = 0; r < o.rows; ++r) {
        if (lifting_ && !written_rows[*tn.out].contains(r)) continue;  // a member row: by its shift
        for (std::uint64_t c = 0; c < o.cols; ++c) {
          const auto& slot = stored.at(*tn.out)[r * per + c];
          const Json where = {{"node", n}, {"op", tn.op}, {"object", o.name}, {"element", {r, c}}};
          if (!slot) refuse("missing_output", "an element of the node's object is not written", where);
          // 1. Leaves.
          const auto expected = materialize(full, r, c);
          const auto actual = leaves(*slot);
          if (actual != expected) {
            auto d = where;
            d["expected"] = render(expected);
            d["actual"] = render(actual);
            refuse("leaves", "the output's leaf set differs from the term's operands", d);
          }
          // 2. Primitive.
          if (tn.primitive) {
            const auto x = resolve(*slot);
            const auto* b = x.conversion() ? nullptr : dynamic_cast<const v::BinaryOp*>(&g_.definingNode(x).operation());
            static const std::map<std::string, v::BinaryOpcode> opcodes{
                {"add", v::BinaryOpcode::Add}, {"sub", v::BinaryOpcode::Subtract},
                {"mul", v::BinaryOpcode::Multiply}, {"max", v::BinaryOpcode::Maximum}, {"min", v::BinaryOpcode::Minimum}};
            const auto want = opcodes.find(tn.operation);
            if (want == opcodes.end()) unsupported("primitive_operation", "no PTX opcode for " + tn.operation);
            if (!b || b->opcode() != want->second || g_.type(x).element().toString() != tn.type || tn.args.size() != 2) {
              auto d = where;
              d["found"] = std::string(g_.definingNode(x).operation().name()) + " " + g_.type(x).toString();
              refuse("primitive", "the stored value is not the term's primitive operation", d);
            }
            const auto operands = g_.instanceOperands(x);
            std::array<std::optional<std::uint64_t>, 2> want_leaf;
            for (std::size_t k = 0; k < 2; ++k) {
              const auto& a = tn.args[k];
              if (a.body || a.reads.size() != 1 || a.reads[0][0].kind == Index::All || a.reads[0][1].kind == Index::All)
                unsupported("primitive_operand", "a primitive operand that is not one input element", where);
              const auto s = materialize({{t_.inputs[a.ref], a.reads[0]}}, r, c);
              want_leaf[k] = s.front().begin;
            }
            const auto match = [&](std::size_t i, std::size_t j) {
              const auto& e0 = t_.objects[t_.inputs[tn.args[0].ref]].element;
              const auto& e1 = t_.objects[t_.inputs[tn.args[1].ref]].element;
              return operandLeaf(operands[i], tn.args[0].convert, e0) == want_leaf[0] &&
                     operandLeaf(operands[j], tn.args[1].convert, e1) == want_leaf[1];
            };
            if (!match(0, 1) && !(tn.commutative && match(1, 0)))
            {
              auto d = where;
              d["found"] = describe(x);
              refuse("primitive_operands", "the primitive's operands are not the term's elements in their formats", d);
            }
          }
          // 3. Factorization.
          if (has_body) {
            const auto expected_cut = materialize(cut, r, c);
            const auto actual_cut = cutter.leaves(*slot);
            if (actual_cut != expected_cut) {
              auto d = where;
              d["expected"] = render(expected_cut);
              d["actual"] = cutter.render(actual_cut);
              refuse("factorization", "with the inner nodes' stored values as leaves, the output reaches other leaves", d);
            }
          }
          ++checked;
        }
      }
      nodes.push_back({{"node", n}, {"op", tn.op}, {"atom", tn.primitive ? "primitive" : "opaque"},
                       {"object", o.name}, {"elements", checked},
                       {"checks", has_body ? Json{"leaves", tn.primitive ? "primitive" : "-", "factorization"}
                                           : Json{"leaves", tn.primitive ? "primitive" : "-"}}});
    }
    return {{"result", "matched"}, {"nodes", nodes}, {"class_mode", class_mode}, {"output_writes", writes}, {"memoized_values", memo_.size()}};
  }

 private:
  const v::Graph& g_;
  const MemoryEliminationResult& m_;
  const Term& t_;
  const UnitLifting* lifting_;
  ScalarSpecialization spec_;
  std::map<Key, std::uint64_t> stop_;
  std::map<Key, SetRef> memo_;
  std::vector<std::string> foreign_;
};

}  // namespace

std::string checkStructuralTerm(const value_ir::Graph& graph, const MemoryEliminationResult& memory,
                                const std::string& term_json, const UnitLifting* lifting) {
  Json report{{"schema", "sentryward.structural-report/v1"}};
  try {
    const auto term = parseTerm(Json::parse(term_json));
    report["row"] = term.row;
    if (!memory.complete) unsupported("memory_elimination", "memory elimination is not complete");
    Checker checker(graph, memory, term, lifting);
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
