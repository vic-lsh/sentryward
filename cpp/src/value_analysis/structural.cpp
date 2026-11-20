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
#include "sentryward/value_ir/ptx_operation.h"
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
  std::uint64_t bytes = 0, stride = 0, rows = 0, cols = 0, base = 0;
  bool row_indexed = true;  // rows follow the launch's row space (else a fixed table, e.g. a weight)
};
// A read coordinate: output coordinate `value` (Out), all of [0, value) (All), `value` (Const), or,
// for a per-segment read, relative to output coordinate `value` and a segment size: the output's
// segment [floor(o / size) * size, + size) (Block) or the output's offset in it, o % size (Mod).
struct Index {
  enum Kind { Out, All, Const, Block, Mod } kind;
  std::uint64_t value;
  std::uint64_t size = 0;
};
using Pattern = std::array<Index, 2>;  // (row, column)
struct Arg {
  bool body = false;
  std::size_t ref = 0;  // term input (an object) or an earlier term node
  std::string convert;  // the element type a primitive operand is converted to; empty: none
  std::vector<Pattern> reads;
};
// A role of an opaque atom's argument: a structural fact on the dataflow graph that tells
// equal-shaped operands apart. An argument carries a set of roles (empty: none).
// through_transcendental: every element reaches the output by some path through ex2, lg2, sin,
//   cos or tanh. avoids_transcendental: by no such path.
// addend: every path to the output passes only adds, an fma's addend, conversions, and bit
//   operations with constant operands (bf16 widening), and no add's other operand also reaches
//   the element.
// multiplicand: every element reaches the output as a factor of a multiply (or an fma's
//   product) whose other factor reaches no element of the argument.
enum class RoleKind { ThroughTranscendental, AvoidsTranscendental, Addend, Multiplicand };
using Role = RoleKind;
struct TermNode {
  std::string op;
  bool primitive = false;
  std::string operation, type;  // type: the format clause's compute element
  std::string store;            // the out port's element the result is converted to; empty: none
  bool commutative = false;
  std::vector<std::vector<Role>> roles;  // opaque: a role set per argument
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
  if (j.at("schema") != "sentryward.structural-term/v2") throw std::invalid_argument("term: schema");
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
    x.base = o.at("base_bytes");
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
    const auto& format = n.at("format");
    x.type = format.at("compute").is_null() ? "" : std::string(format.at("compute"));
    if (!format.at("store").is_null()) x.store = format.at("store");
    if (x.primitive) {
      x.operation = atom.at("operation");
      x.commutative = atom.at("commutative");
    } else if (atom.at("kind") == "opaque") {
      for (const auto& set : atom.at("roles")) {
        if (!set.is_array()) throw std::invalid_argument("term: a role set is an array");
        std::vector<Role> rs;
        for (const auto& r : set) {
          if (r == "through_transcendental") rs.push_back(RoleKind::ThroughTranscendental);
          else if (r == "avoids_transcendental") rs.push_back(RoleKind::AvoidsTranscendental);
          else if (r == "addend") rs.push_back(RoleKind::Addend);
          else if (r == "multiplicand") rs.push_back(RoleKind::Multiplicand);
          else throw std::invalid_argument("term: role " + r.dump());
        }
        x.roles.push_back(std::move(rs));
      }
    } else {
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
          else if (c.contains("block")) q[d] = {Index::Block, c.at("block").at("out"), c.at("block").at("size")};
          else if (c.contains("mod")) q[d] = {Index::Mod, c.at("mod").at("out"), c.at("mod").at("size")};
          else throw std::invalid_argument("term: coordinate");
          const bool rel = q[d].kind == Index::Out || q[d].kind == Index::Block || q[d].kind == Index::Mod;
          if (rel && q[d].value > 1) throw std::invalid_argument("term: output coordinate");
          if ((q[d].kind == Index::Block || q[d].kind == Index::Mod) && q[d].size == 0)
            throw std::invalid_argument("term: segment size");
        }
        y.reads.push_back(q);
      }
      x.args.push_back(std::move(y));
    }
    if (n.contains("out") && !n.at("out").is_null()) x.out = objectIndex(t, n.at("out"));
    if (!x.primitive && x.roles.size() != x.args.size()) throw std::invalid_argument("term: one role per argument");
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
  for (auto& c : r) {
    if (c.kind == Index::Out) {
      c = outer[c.value];
    } else if (c.kind == Index::Block || c.kind == Index::Mod) {
      // A segment of a body value read at the output's own coordinate; anything else is not
      // stated by these coordinates.
      if (outer[c.value].kind != Index::Out) throw std::invalid_argument("term: a segment read through a non-identity read");
      c.value = outer[c.value].value;
    }
  }
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
        else if (c.kind == Index::Block) lo[d] = at[c.value] / c.size * c.size, hi[d] = lo[d] + c.size;
        else if (c.kind == Index::Mod) lo[d] = at[c.value] % c.size, hi[d] = lo[d] + 1;
        else lo[d] = 0, hi[d] = c.value;
      }
      const auto per = o.stride / o.bytes;
      for (auto r = lo[0]; r < hi[0]; ++r)
        xs.push_back({(p.space << kShift) | (r * per + lo[1]), (p.space << kShift) | (r * per + hi[1])});
    }
    return normalize(std::move(xs));
  }

  // The one coordinate pattern `at` names at output element (row, col), or none when it names a
  // range (All, Block).
  static std::optional<std::array<std::uint64_t, 2>> coordinate(const Pattern& at, std::uint64_t row, std::uint64_t col) {
    const std::array<std::uint64_t, 2> out{row, col};
    std::array<std::uint64_t, 2> x{};
    for (std::size_t d = 0; d < 2; ++d) {
      const auto& c = at[d];
      if (c.kind == Index::Out) x[d] = out[c.value];
      else if (c.kind == Index::Const) x[d] = c.value;
      else if (c.kind == Index::Mod) x[d] = out[c.value] % c.size;
      else return std::nullopt;
    }
    return x;
  }

  // The term object holding byte `offset` of allocation `allocation` and that byte's element
  // index (row * (stride / bytes) + column, from the object's base), or none.
  std::optional<std::pair<std::size_t, std::uint64_t>> elementAt(const std::string& allocation,
                                                                 std::uint64_t offset) const {
    for (std::size_t i = 0; i < t_.objects.size(); ++i) {
      const auto& o = t_.objects[i];
      if (o.allocation != allocation || offset < o.base) continue;
      const auto rel = offset - o.base, row = rel / o.stride, col = rel % o.stride / o.bytes;
      if (row < o.rows && col < o.cols) return std::pair{i, row * (o.stride / o.bytes) + col};
    }
    return std::nullopt;
  }

  // Leaves of the bytes [offset, offset + bytes) of allocation `allocation`'s initial contents:
  // the elements of the term objects they lie in; bytes in no term object are leaves of a space
  // per allocation (never a match).
  Set regionLeaves(const std::string& allocation, std::uint64_t offset, std::uint64_t bytes) {
    std::vector<Interval> xs;
    for (std::uint64_t b = offset; b < offset + bytes;) {
      if (const auto e = elementAt(allocation, b)) {
        const auto& o = t_.objects[e->first];
        xs.push_back({(std::uint64_t(e->first) << kShift) | e->second, ((std::uint64_t(e->first) << kShift) | e->second) + 1});
        b = b - (b - o.base) % o.bytes + o.bytes;
        continue;
      }
      auto it = std::find(foreign_.begin(), foreign_.end(), allocation);
      const auto space = t_.objects.size() + t_.nodes.size() + std::size_t(it - foreign_.begin());
      if (it == foreign_.end()) foreign_.push_back(allocation);
      xs.push_back({(space << kShift) | b, ((space << kShift) | b) + 1});
      ++b;
    }
    return normalize(std::move(xs));
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

  static bool transcendental(const v::Operation& op) {
    if (dynamic_cast<const v::IdealExp2Op*>(&op)) return true;
    const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op);
    if (!p) return false;
    using O = v::PtxScalarOpcode;
    const auto c = p->opcode();
    return c == O::Exp2 || c == O::Log2 || c == O::Sin || c == O::Cos || c == O::Tanh;
  }

  // The leaves x reaches by a path through a transcendental operation (memoized).
  Set transLeaves(v::Value x, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    if (x.threadSelector()) unsupported("thread_selector", "a thread selector the launch does not decide");
    if (x.conversion()) return transLeaves(x.conversion()->source, depth + 1);
    const Key key{x.nodeId(), x.resultIndex(), x.instanceIndex()};
    const bool plain = x.indices().empty() && !x.pointer();
    // A cut value (an inner node's bound output) ends the path: what lies below it is the
    // producer's, whose own roles cover its inputs.
    if (plain && stop_.count(key)) return {};
    if (plain)
      if (auto it = trans_memo_.find(key); it != trans_memo_.end()) return *it->second;
    const auto& node = g_.definingNode(x);
    Set s;
    if (transcendental(node.operation())) {
      s = leaves(x, depth + 1);
    } else if (dynamic_cast<const v::LoadOp*>(&node.operation())) {
      if (const auto* r = m_.readReplacement({x.nodeId(), x.instanceIndex()}))
        if (const auto* value = std::get_if<v::Value>(r)) s = transLeaves(*value, depth + 1);
    } else {
      for (const auto& a : g_.instanceOperands(x)) s = unite(s, transLeaves(a, depth + 1));
    }
    if (plain) trans_memo_.emplace(key, std::make_shared<const Set>(s));
    return s;
  }

  // The expected leaves of argument k of term node m when m is evaluated at outer pattern `at`
  // (identity for a node at its own output), at output element (r, c).
  Set argLeaves(std::size_t m, std::size_t k, const Pattern& at, std::uint64_t r, std::uint64_t c) const {
    const auto& a = t_.nodes[m].args[k];
    std::vector<SpacePattern> ps;
    for (const auto& p : a.reads) {
      const auto here = compose(p, at);
      if (!a.body) ps.push_back({t_.inputs[a.ref], here});
      else
        for (const auto& q : patterns(t_, a.ref, false)) ps.push_back({q.space, compose(q.at, here)});
    }
    return materialize(ps, r, c);
  }

  // Memoized per argument set k (roles are few, so one memo per (root, k)).
  using Memo = std::map<Key, SetRef>;

  // The leaves of k that x reaches as a factor of a multiply (or an fma's product) whose other
  // factor reaches no leaf of k.
  Set factorLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, unsigned d) {
      Set s;
      const auto* b = dynamic_cast<const v::BinaryOp*>(&op);
      const bool product = (b && b->opcode() == v::BinaryOpcode::Multiply) || dynamic_cast<const v::FmaOp*>(&op);
      if (product && xs.size() >= 2) {
        const auto l0 = meet(leaves(xs[0], d + 1), k), l1 = meet(leaves(xs[1], d + 1), k);
        if (l1.empty()) s = unite(s, l0);
        if (l0.empty()) s = unite(s, l1);
      }
      for (const auto& a : xs) s = unite(s, factorLeaves(a, k, memo, d + 1));
      return s;
    });
  }

  // The leaves of k that x reaches on some path that passes no product whose other factor reaches
  // no leaf of k (a path that bypasses the multiply).
  Set bypassLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, unsigned d) {
      const auto* b = dynamic_cast<const v::BinaryOp*>(&op);
      const bool product = (b && b->opcode() == v::BinaryOpcode::Multiply) || dynamic_cast<const v::FmaOp*>(&op);
      Set s;
      std::size_t from = 0;
      if (product && xs.size() >= 2) {
        const auto l0 = meet(leaves(xs[0], d + 1), k), l1 = meet(leaves(xs[1], d + 1), k);
        // The covered factor's paths pass the product; the other factor's are its own.
        if (l1.empty() && !l0.empty()) s = unite(s, bypassLeaves(xs[1], k, memo, d + 1));
        else if (l0.empty() && !l1.empty()) s = unite(s, bypassLeaves(xs[0], k, memo, d + 1));
        else for (std::size_t i = 0; i < 2; ++i) s = unite(s, operandBypass(xs[i], k, memo, d + 1));
        from = 2;  // an fma's addend is not a factor
      }
      for (std::size_t i = from; i < xs.size(); ++i) s = unite(s, operandBypass(xs[i], k, memo, d + 1));
      return s;
    });
  }
  // An operand's bypass leaves: a leaf of k itself, else its own walk.
  Set operandBypass(const v::Value& a, const Set& k, Memo& memo, unsigned depth) {
    auto y = resolve(a);
    while (y.conversion()) y = resolve(y.conversion()->source);
    if (const auto* ld = dynamic_cast<const v::LoadOp*>(&g_.definingNode(y).operation())) {
      (void)ld;
      const auto* rep = m_.readReplacement({y.nodeId(), y.instanceIndex()});
      if (!rep || !std::get_if<v::Value>(rep)) return meet(leaves(a, depth), k);
    }
    return bypassLeaves(a, k, memo, depth);
  }

  // The leaves of k that x reaches by some path other than adds, an fma's addend, conversions and
  // bit operations with constant operands, or as both operands of one add.
  Set nonAddendLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, unsigned d) {
      Set s;
      const auto* b = dynamic_cast<const v::BinaryOp*>(&op);
      const auto code = b ? std::optional(b->opcode()) : std::nullopt;
      using B = v::BinaryOpcode;
      if (code == B::Add && xs.size() == 2) {
        s = meet(meet(leaves(xs[0], d + 1), leaves(xs[1], d + 1)), k);
        for (const auto& a : xs) s = unite(s, nonAddendLeaves(a, k, memo, d + 1));
      } else if (dynamic_cast<const v::FmaOp*>(&op) && xs.size() == 3) {
        const auto p = unite(leaves(xs[0], d + 1), leaves(xs[1], d + 1));
        s = unite(meet(p, k), meet(meet(p, leaves(xs[2], d + 1)), k));
        s = unite(s, nonAddendLeaves(xs[2], k, memo, d + 1));
      } else {
        // A bit operation whose other operands are constants (no leaves) carries its one operand.
        std::vector<std::size_t> live;
        for (std::size_t i = 0; i < xs.size(); ++i)
          if (!leaves(xs[i], d + 1).empty()) live.push_back(i);
        const bool bits = code == B::ShiftLeft || code == B::ShiftRight || code == B::BitAnd || code == B::BitOr;
        if (bits && live.size() == 1) return nonAddendLeaves(xs[live[0]], k, memo, d + 1);
        for (const auto& a : xs) s = unite(s, meet(leaves(a, d + 1), k));
      }
      return s;
    });
  }

  // A memoized walk of x's defining DAG: through conversions and replaced loads, `f` on any other
  // operation with its operands; an unreplaced load (a leaf) gives the empty set.
  template <class F>
  Set walk(v::Value x, Memo& memo, unsigned depth, const F& f) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    if (x.threadSelector()) unsupported("thread_selector", "a thread selector the launch does not decide");
    if (x.conversion()) return walk(x.conversion()->source, memo, depth + 1, f);
    const Key key{x.nodeId(), x.resultIndex(), x.instanceIndex()};
    const bool plain = x.indices().empty() && !x.pointer();
    if (plain)
      if (auto it = memo.find(key); it != memo.end()) return *it->second;
    const auto& op = g_.definingNode(x).operation();
    Set s;
    if (dynamic_cast<const v::LoadOp*>(&op)) {
      if (const auto* r = m_.readReplacement({x.nodeId(), x.instanceIndex()}))
        if (const auto* value = std::get_if<v::Value>(r)) s = walk(*value, memo, depth + 1, f);
    } else {
      s = f(op, g_.instanceOperands(x), depth);
    }
    if (plain) memo.emplace(key, std::make_shared<const Set>(s));
    return s;
  }

  // The cut elements of body argument k of node m (the producer's bound output elements it reads)
  // at outer pattern `at`, output element (r, c).
  Set argCut(std::size_t m, std::size_t k, const Pattern& at, std::uint64_t r, std::uint64_t c) const {
    const auto& a = t_.nodes[m].args[k];
    std::vector<SpacePattern> ps;
    for (const auto& p : a.reads) ps.push_back({t_.objects.size() + a.ref, compose(p, at)});
    return materialize(ps, r, c);
  }

  // Role sets of opaque node m over its value `root` (m at outer pattern `at`). A transcendental
  // role is a fact about the paths from the argument's value to m's output: for an input argument
  // from its loads, for a body argument from the node bound to its producer's output, so those
  // roles run on `boundary` (this term's cut checker: inner nodes' stored values are leaves and
  // end each path). A body argument with a transcendental role and no stored, bound output fails.
  // Addend and Multiplicand keep their independence condition over the argument's leaves.
  void checkRoles(std::size_t m, const v::Value& root, const Pattern& at, std::uint64_t r, std::uint64_t c,
                  const Json& where, Checker* boundary = nullptr) {
    const auto& tn = t_.nodes[m];
    if (tn.primitive) return;
    // Reached-through-transcendental sets: from the loads (input arguments: every path from the
    // argument's loads, through producers too) and on the cut checker (body arguments).
    std::optional<Set> through_in, through_cut;
    for (std::size_t k = 0; k < tn.args.size(); ++k) {
      if (tn.roles[k].empty()) continue;
      const auto& arg = tn.args[k];
      const auto e = argLeaves(m, k, at, r, c);
      for (const auto role : tn.roles[k]) {
        Set reached;
        bool ok = false;
        const char* name = "";
        Memo memo;
        switch (role) {
          case RoleKind::ThroughTranscendental:
          case RoleKind::AvoidsTranscendental: {
            name = role == RoleKind::ThroughTranscendental ? "through_transcendental" : "avoids_transcendental";
            if (arg.body && (!boundary || !t_.nodes[arg.ref].out)) {
              auto d = where;
              d["argument"] = k;
              d["role"] = name;
              refuse("role_boundary", "a body argument's role needs its producer's output bound to a stored node", d);
            }
            const auto ek = arg.body ? argCut(m, k, at, r, c) : e;
            auto& through = arg.body ? through_cut : through_in;
            if (!through) through = arg.body ? boundary->transLeaves(root) : transLeaves(root);
            reached = *through;
            ok = role == RoleKind::ThroughTranscendental ? unite(reached, ek) == reached : disjoint(reached, ek);
            break;
          }
          case RoleKind::Addend:
            reached = nonAddendLeaves(root, e, memo);
            ok = reached.empty();
            name = "addend";
            break;
          case RoleKind::Multiplicand: {
            // Every leaf of the argument is reached as a factor of a product whose other factor
            // is independent of it, and no path reaches one without passing such a product.
            reached = factorLeaves(root, e, memo);
            Memo bmemo;
            const auto bypass = bypassLeaves(root, e, bmemo);
            ok = reached == e && bypass.empty();
            if (!bypass.empty()) reached = bypass;
            name = "multiplicand";
            break;
          }
        }
        if (!ok) {
          auto d = where;
          d["argument"] = k;
          d["role"] = name;
          d["argument_leaves"] = render(e);
          d["reached"] = render(reached);
          refuse("role", "an argument's role signature does not hold", d);
        }
      }
    }
  }

  // The intersection of two sorted range sets.
  static Set meet(const Set& a, const Set& b) {
    Set s;
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
      const auto lo = std::max(a[i].begin, b[j].begin), hi = std::min(a[i].end, b[j].end);
      if (lo < hi) s.push_back({lo, hi});
      (a[i].end < b[j].end) ? ++i : ++j;
    }
    return s;
  }

  static bool disjoint(const Set& a, const Set& b) {
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
      if (a[i].end <= b[j].begin) ++i;
      else if (b[j].end <= a[i].begin) ++j;
      else return false;
    }
    return true;
  }

  // Class mode: the representatives' written elements, and each member's as a representative's
  // shifted by s elements; together they must cover every element of every output object once.
  // A member's shift, from its per-allocation offsets (fixed objects: zero; an allocation the
  // kernel does not access constrains nothing):
  // - whole rows: every accessed row-indexed object moves by the same k rows (s = k rows); the
  //   term must read row-indexed objects at the output's row only, so its leaves shift with k;
  // - elements: every accessed row-indexed object moves by the same e elements (s = e); sound only
  //   for an elementwise term (every read of a row-indexed object at the output's element) over
  //   contiguous rows of one width, where (row, column) is the linear element index.
  // Null without lifting.
  Json classMode(const std::map<std::size_t, std::set<std::uint64_t>>& written,
                 const std::map<std::uint64_t, std::set<std::uint64_t>>& by_cta) {
    if (!lifting_) return nullptr;
    std::optional<std::set<std::uint64_t>> rep;
    std::optional<std::pair<std::uint64_t, std::uint64_t>> shape;  // rows, elements per row
    std::uint64_t cols = 0;
    for (const auto& n : t_.nodes) {
      if (!n.out) continue;
      const auto& o = t_.objects[*n.out];
      const auto it = written.find(*n.out);
      const auto here = it == written.end() ? std::set<std::uint64_t>{} : it->second;
      if (rep && *rep != here) refuse("class_rows", "output objects' representative elements differ");
      rep = here;
      const std::pair<std::uint64_t, std::uint64_t> sh{o.rows, o.stride / o.bytes};
      if ((shape && (*shape != sh || cols != o.cols)) || !o.row_indexed)
        unsupported("class_mode_rows", "output objects with different row spaces or widths");
      shape = sh;
      cols = o.cols;
    }
    if (!shape) return nullptr;
    const auto [rows, per] = *shape;
    bool row_local = true, elementwise = true;
    for (std::size_t n = 0; n < t_.nodes.size(); ++n)
      for (const bool cut : {false, true})
        for (const auto& p : patterns(t_, n, cut)) {
          if (!layout(p.space).row_indexed) continue;
          row_local &= p.at[0].kind == Index::Out && p.at[0].value == 0;
          elementwise &= row_local && p.at[1].kind == Index::Out && p.at[1].value == 1;
        }
    for (const auto& o : t_.objects)
      if (o.row_indexed) elementwise &= o.stride == o.cols * o.bytes && o.cols == cols;
    // Per-segment term (segments of `seg` columns): row-indexed objects read at the output's row
    // and at its column or its segment, fixed objects at no output column but the offset in the
    // segment (mod), every row-indexed object as wide as the output. A shift of whole rows plus
    // whole segments within the row then moves every leaf with the output.
    std::optional<std::uint64_t> seg;
    bool segmented = row_local;
    for (std::size_t n = 0; n < t_.nodes.size(); ++n)
      for (const bool cut : {false, true})
        for (const auto& p : patterns(t_, n, cut)) {
          const auto& c = p.at[1];
          if (c.kind == Index::Block || c.kind == Index::Mod) {
            segmented &= !seg || *seg == c.size;
            seg = c.size;
          }
          if (layout(p.space).row_indexed) segmented &= c.kind == Index::Out || c.kind == Index::Block;
          else segmented &= p.at[0].kind != Index::Out && (c.kind == Index::Const || c.kind == Index::All || c.kind == Index::Mod);
        }
    for (const auto& o : t_.objects)
      if (o.row_indexed) segmented &= o.cols == cols;
    segmented &= seg.has_value();
    const auto& L = *lifting_;
    std::vector<bool> covered(rows * per, false);
    std::size_t members = 0, member_elements = 0;
    const auto cover = [&](std::int64_t e) {
      if (e < 0 || std::uint64_t(e) >= rows * per || std::uint64_t(e) % per >= cols || covered[std::size_t(e)])
        refuse("class_coverage", "an element covered twice or outside the launch", {{"element", e}});
      covered[std::size_t(e)] = true;
    };
    std::map<UnitLifting::Origin, std::set<std::uint64_t>> by_origin;
    for (const auto& [cta, es] : by_cta) {
      const UnitLifting::Origin c{cta % L.grid[0], cta / L.grid[0] % L.grid[1], cta / (L.grid[0] * L.grid[1])};
      UnitLifting::Origin o;
      for (std::size_t a = 0; a < 3; ++a) o[a] = c[a] / L.unit_shape[a] * L.unit_shape[a];
      by_origin[o].insert(es.begin(), es.end());
    }
    std::set<std::string> modes;
    for (const auto& [origin, es] : by_origin) {
      for (const auto e : es) cover(std::int64_t(e));
      const auto it = L.members.find(origin);
      if (it == L.members.end()) continue;
      for (const auto& m : it->second) {
        for (const auto& [alloc, d] : m.delta) {
          bool known = false;
          for (const auto& o : t_.objects) known |= o.allocation == alloc;
          if (!known && d) refuse("class_shift", "a member moves an allocation the term does not name", {{"allocation", alloc}});
        }
        std::optional<std::int64_t> krows, kelems;
        std::optional<std::pair<std::int64_t, std::int64_t>> kseg;  // rows, columns
        bool whole = true, linear = true, segs = segmented;
        for (const auto& o : t_.objects) {
          if (!m.delta.contains(o.allocation)) continue;
          const auto d = m.delta.at(o.allocation);
          if (!o.row_indexed) {
            if (d) refuse("class_shift", "a member moves a fixed object", {{"object", o.name}, {"delta", d}});
            continue;
          }
          const auto k = d / std::int64_t(o.stride), e = d / std::int64_t(o.bytes);
          whole &= d % std::int64_t(o.stride) == 0 && (!krows || *krows == k);
          if (segs) {
            const auto st = std::int64_t(o.stride);
            const auto kr = (d >= 0 ? d : d - st + 1) / st, rem = d - kr * st;
            const auto j = rem / std::int64_t(o.bytes);
            segs = rem % std::int64_t(o.bytes) == 0 && j % std::int64_t(*seg) == 0 && j < std::int64_t(o.cols) &&
                   (!kseg || *kseg == std::pair{kr, j});
            kseg = std::pair{kr, j};
          }
          linear &= d % std::int64_t(o.bytes) == 0 && (!kelems || *kelems == e);
          krows = k;
          kelems = e;
        }
        if (!krows) unsupported("class_shift", "no row-indexed object");
        std::int64_t shift;
        if (whole && row_local) {
          shift = *krows * std::int64_t(per);
          modes.insert("rows");
        } else if (linear && elementwise) {
          shift = *kelems;
          modes.insert("elements");
        } else if (segs && kseg) {
          shift = kseg->first * std::int64_t(per) + kseg->second;
          modes.insert("segments");
        } else {
          refuse("class_shift", whole ? "whole-row shifts but the term reads other rows"
                                      : "a member's offsets are no common row or (for an elementwise term) element shift");
        }
        ++members;
        for (const auto e : es) {
          cover(std::int64_t(e) + shift);
          ++member_elements;
        }
      }
    }
    for (std::uint64_t e = 0; e < rows * per; ++e)
      if (e % per < cols && !covered[e]) refuse("class_coverage", "an element no representative or member covers", {{"element", e}});
    return {{"representatives", by_origin.size()}, {"representative_elements", rep ? rep->size() : 0},
            {"members", members}, {"member_elements", member_elements}, {"shifts", modes}};
  }

  Json run() {
    // Stored values per output object element.
    std::map<std::size_t, std::vector<std::optional<v::Value>>> stored;
    for (const auto& n : t_.nodes)
      if (n.out) stored[*n.out].assign(t_.objects[*n.out].rows * (t_.objects[*n.out].stride / t_.objects[*n.out].bytes), std::nullopt);
    std::size_t writes = 0;
    std::map<std::size_t, std::set<std::uint64_t>> written;
    std::map<std::uint64_t, std::set<std::uint64_t>> by_cta;
    m_.forEachOutput(g_, [&](const OutputWrite& w) {
      ++writes;
      const auto at = elementAt(w.region.allocation, w.region.offset);
      if (!at || !stored.contains(at->first))
        refuse("unexpected_output", "a write outside every term node's output object",
               {{"allocation", w.region.allocation}, {"offset", w.region.offset}});
      const auto obj = at->first;
      const auto& o = t_.objects[obj];
      if (w.reduced || !w.base.empty())
        unsupported("output_form", "a reduced or symbolically placed output", {{"object", o.name}});
      if (w.region.bytes != o.bytes || (w.region.offset - o.base) % o.bytes || w.value_byte_offset ||
          !g_.type(w.value).isScalar())
        unsupported("output_granularity", "an output write that is not one element of its object",
                    {{"object", o.name}, {"offset", w.region.offset}, {"bytes", w.region.bytes}});
      const auto e = at->second;
      auto& slot = stored[obj][e];
      if (slot) refuse("duplicate_output", "two final writes to one element", {{"object", o.name}, {"offset", w.region.offset}});
      slot = w.value;
      written[obj].insert(e);
      const auto cta = g_.node(w.writer.node).instanceOrigin(w.writer.instance).cta;
      if (lifting_ && !cta) unsupported("writer_cta", "an output write without a writer CTA");
      by_cta[cta.value_or(0)].insert(e);
    });
    const auto class_mode = classMode(written, by_cta);
    Json nodes = Json::array();
    for (std::size_t n = 0; n < t_.nodes.size(); ++n) {
      const auto& tn = t_.nodes[n];
      if (!tn.out) {
        bool searched = false;
        for (const auto& u : t_.nodes)
          for (const auto& a : u.args) searched |= a.body && a.ref == n && !u.primitive;
        if (searched)
          unsupported("inner_without_output", "an inner term node without an output object under an opaque atom (cut search not built)", {{"node", n}, {"op", tn.op}});
        continue;  // its value is the operand of each primitive user, checked there
      }
      const auto& o = t_.objects[*tn.out];
      const auto per = o.stride / o.bytes;
      const auto full = patterns(t_, n, false), cut = patterns(t_, n, true);
      const bool has_body = std::any_of(tn.args.begin(), tn.args.end(), [](const Arg& a) { return a.body; });
      // The cut leaves: each inner node's stored values, by element.
      std::map<Key, std::uint64_t> stop;
      if (has_body && !tn.primitive)
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
        for (std::uint64_t c = 0; c < o.cols; ++c) {
          const auto& slot = stored.at(*tn.out)[r * per + c];
          if (lifting_ && !slot) continue;  // a member's element: covered by its shift (classMode)
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
          // 2. Primitive: the stored value (less the format clause's store conversion) is the one
          // operation on the operands, up to commutativity: an input element through its
          // conversion, an inner node's stored value, or an inner node's value (the cut) whose leaves
          // are that node's operands there.
          if (tn.primitive) {
            auto x = resolve(*slot);
            auto d = where;
            d["found"] = describe(x);
            // The store: the op's result converted to the out port's element, or already of it; a
            // same-width bitcast (a BF16 element stored as u16) is representation.
            while (x.conversion() && x.conversion()->spec->kind == v::TypeConversionKind::Bitcast &&
                   g_.type(x.conversion()->source).element().width() == g_.type(x).element().width() &&
                   x.indices().empty())
              x = resolve(x.conversion()->source);
            if (!tn.store.empty()) {
              if (x.conversion() && x.conversion()->spec->kind == v::TypeConversionKind::Convert &&
                  x.conversion()->spec->target.element().toString() == tn.store)
                x = resolve(x.conversion()->source);
              else if (!x.conversion() && dynamic_cast<const v::ConvertOp*>(&g_.definingNode(x).operation()) &&
                       g_.type(x).element().toString() == tn.store)
                x = resolve(g_.instanceOperands(x).at(0));
              else if (x.conversion() || g_.type(x).element().toString() != tn.store)
                refuse("format", "the stored value is not of, or converted to, the out port's element", d);
            }
            const auto& xop = g_.definingNode(x).operation();
            const auto* ptx = dynamic_cast<const v::PtxScalarOp*>(&xop);
            if (!x.conversion() && (tn.operation == "add" || tn.operation == "sub") &&
                (dynamic_cast<const v::FmaOp*>(&xop) || (ptx && ptx->opcode() == v::PtxScalarOpcode::MultiplyAdd)))
              refuse("contraction",
                     "the add is contracted with a product into one fma: no value at the term's atom boundary", d);
            const auto* b = x.conversion() ? nullptr : dynamic_cast<const v::BinaryOp*>(&xop);
            static const std::map<std::string, v::BinaryOpcode> opcodes{
                {"add", v::BinaryOpcode::Add}, {"sub", v::BinaryOpcode::Subtract},
                {"mul", v::BinaryOpcode::Multiply}, {"max", v::BinaryOpcode::Maximum}, {"min", v::BinaryOpcode::Minimum}};
            const auto want = opcodes.find(tn.operation);
            if (want == opcodes.end()) unsupported("primitive_operation", "no PTX opcode for " + tn.operation);
            // The compute element is the op's numerics unless the term names it (D199).
            const auto op_type = g_.type(x).element().toString();
            if (!b || b->opcode() != want->second || tn.args.size() != 2 || g_.type(x).element().kind() != v::ElementKind::Floating ||
                (!tn.type.empty() && op_type != tn.type))
              refuse("primitive", "the stored value is not the term's primitive operation", d);
            // An operand of another element than the op's is converted to the op's.
            const auto convert_to = [&](const std::string& elem, const std::string& named) {
              return !named.empty() ? named : elem == op_type ? std::string() : op_type;
            };
            const auto operands = g_.instanceOperands(x);
            std::vector<std::pair<std::pair<std::size_t, std::uint64_t>, Key>> pending;
            std::optional<std::pair<std::size_t, std::uint64_t>> conflict;
            const auto matches = [&](const v::Value& operand, std::size_t k) -> bool {
              const auto& a = tn.args[k];
              if (a.reads.size() != 1 || a.reads[0][0].kind == Index::All || a.reads[0][1].kind == Index::All || a.reads[0][1].kind == Index::Block)
                unsupported("primitive_operand", "a primitive operand that is not one element", where);
              if (!a.body) {
                const auto s = materialize({{t_.inputs[a.ref], a.reads[0]}}, r, c);
                const auto& elem = t_.objects[t_.inputs[a.ref]].element;
                return operandLeaf(operand, convert_to(elem, a.convert), elem) == s.front().begin;
              }
              auto y = resolve(operand);
              const auto& inner = t_.nodes[a.ref];
              if (inner.out) {
                const auto cv = convert_to(t_.objects[*inner.out].element, a.convert);
                if (!cv.empty()) {
                  if (!(y.conversion() && y.conversion()->spec->target.element().toString() == cv)) return false;
                  y = resolve(y.conversion()->source);
                }
              }
              if (inner.out) {  // the inner node's stored value at the operand's element
                const auto e = materialize({{t_.objects.size() + a.ref, a.reads[0]}}, r, c).front().begin & kElementMask;
                const auto& sv = stored.at(*inner.out).at(e);
                if (!sv) return false;
                // The same value: equal conversion chains (the store's rounding, e.g. to bf16) over
                // the same node result; a bitcast to an integer or bit type keeps the bits and
                // reinterprets no float, so it is skipped on either side (a bf16 stored through its
                // u16 image). A bitcast to a float type is not skipped.
                const auto bits = [&](v::Value v) {
                  while (v.conversion() && v.conversion()->spec->kind == v::TypeConversionKind::Bitcast) {
                    const auto e = v.conversion()->spec->target.element().toString();
                    if (e.empty() || (e[0] != 'u' && e[0] != 's' && e[0] != 'b') || e.rfind("bf", 0) == 0) break;
                    v = resolve(v.conversion()->source);
                  }
                  return v;
                };
                auto z = bits(resolve(*sv));
                y = bits(y);
                while (y.conversion() && z.conversion()) {
                  if (!(*y.conversion()->spec == *z.conversion()->spec)) return false;
                  y = bits(resolve(y.conversion()->source));
                  z = bits(resolve(z.conversion()->source));
                }
                return !y.conversion() && !z.conversion() && y.nodeId() == z.nodeId() &&
                       y.resultIndex() == z.resultIndex() && y.instanceIndex() == z.instanceIndex();
              }
              // The cut: a value whose leaves are the inner node's operands at that element. The
              // inner node's element is one node of the graph: every use binds the same value.
              // An unstored primitive producer is not supported: only its leaves would be checked,
              // not its operation (add(mul(a, b), c) could bind sub(a, b)).
              if (inner.primitive) {
                auto dc = d;
                dc["inner_node"] = a.ref;
                refuse("unstored_primitive_producer", "an unstored primitive inner node: its operation is not checked", dc);
              }
              const auto at = a.reads[0];
              std::vector<SpacePattern> ps;
              for (const auto& q : patterns(t_, a.ref, false)) ps.push_back({q.space, compose(q.at, at)});
              if (leaves(y) != materialize(ps, r, c)) return false;
              // The element's identity is its coordinate (an unstored node has no layout object).
              const auto coord = coordinate(at, r, c);
              if (!coord) unsupported("producer_element", "an unstored inner node read at more than one element", where);
              const std::pair<std::size_t, std::uint64_t> slot_key{a.ref, ((*coord)[0] << 32) | (*coord)[1]};
              const Key value{y.nodeId(), y.resultIndex(), y.instanceIndex()};
              if (auto it = bound_.find(slot_key); it != bound_.end() && !(it->second == value)) {
                conflict = slot_key;
                return false;
              }
              pending.push_back({slot_key, value});
              checkRoles(a.ref, y, at, r, c, where);
              return true;
            };
            const auto attempt = [&](std::size_t i0, std::size_t i1) {
              pending.clear();
              return matches(operands[i0], 0) && matches(operands[i1], 1);
            };
            if (!attempt(0, 1) && !(tn.commutative && attempt(1, 0))) {
              if (conflict) {
                auto dc = d;
                dc["inner_node"] = conflict->first;
                dc["inner_element"] = conflict->second;
                refuse("producer_binding", "an unstored inner node's element binds two values at two uses", dc);
              }
              refuse("primitive_operands", "the primitive's operands are not the term's operands in their formats", d);
            }
            for (const auto& [k, v] : pending) {
              if (auto [it, added] = bound_.emplace(k, v); !added && !(it->second == v)) {
                auto dc = d;
                dc["inner_node"] = k.first;
                dc["inner_element"] = k.second;
                refuse("producer_binding", "an unstored inner node's element binds two values at two uses", dc);
              }
            }
          } else {
            checkRoles(n, *slot, Pattern{Index{Index::Out, 0}, Index{Index::Out, 1}}, r, c, where,
                       has_body ? &cutter : nullptr);
          }
          // 3. Factorization.
          if (has_body && !tn.primitive) {
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
                       {"checks", tn.primitive ? Json{"leaves", "primitive"}
                                               : has_body ? Json{"leaves", "roles", "factorization"}
                                                          : Json{"leaves", "roles"}}});
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
  // Per unstored inner node and element: the one value its uses bind (`producer_binding`).
  std::map<std::pair<std::size_t, std::uint64_t>, Key> bound_;
  std::map<Key, SetRef> memo_;
  std::map<Key, SetRef> trans_memo_;
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
