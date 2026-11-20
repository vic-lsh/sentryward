// Structural mode: see structural.h and docs/structural-mode.md.
#include "sentryward/value_analysis/structural.h"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "sentryward/equivalence/constant_values.h"
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
  std::optional<Set> loadLeaves(const v::Value& x, bool initial_only, bool full = false) {
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
        return full ? fullLeaves(source) : leaves(source);
      }
      if (const auto* initial = std::get_if<InitialMemorySymbol>(r))
        return narrow(initial->region.allocation, initial->region.offset, initial->region.bytes);
      if (initial_only) return std::nullopt;
      if (!x.indices().empty()) unsupported("load_component", "a component of a reassembled read");
      const auto piece = [&](const ReinterpretedRead& p) -> Set {
        if (const auto* s = std::get_if<v::Value>(&p.source)) {
          // A piece of a stored value: exact only when it is all of the value's bits (a same-width
          // reinterpretation keeps every bit, so every leaf). A part of the value depends on the
          // value's bytes, which this check does not track: crediting all its leaves could hide a
          // missing dependency (a pack of a's low half and b's high half, read back as the low
          // half, depends on a only).
          if (p.byte_offset != 0 || p.type.bitSize() != g_.type(*s).bitSize())
            refuse("partial_stored_read", "a read of part of a stored value (its per-byte dependencies are not decided)",
                   {{"node", key.node}, {"instance", key.instance}, {"byte_offset", p.byte_offset},
                    {"bits", p.type.bitSize()}, {"stored_bits", g_.type(*s).bitSize()}});
          return full ? fullLeaves(*s) : leaves(*s);
        }
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

  // The identity of value x, as a canonical key: its node result and instance, pointer properties,
  // indices and conversion chain, each level resolved. Two values are one value iff their keys are
  // equal; every comparison or map of values in this check uses it (cut values, producer bindings,
  // primitive operands, memos). A thread selector the launch does not decide is unsupported.
  std::string ident(v::Value x, unsigned depth = 0) {
    if (depth > 512) unsupported("identity_depth", "a value's conversion and index nesting is deeper than 512");
    x = resolve(x);
    if (x.threadSelector()) unsupported("thread_selector", "a thread selector the launch does not decide");
    std::string s = std::to_string(x.graphIdentity()) + ":" + std::to_string(x.nodeId()) + "." +
                    std::to_string(x.resultIndex()) + "." + std::to_string(x.instanceIndex());
    if (const auto& c = x.conversion()) {
      const auto& sp = *c->spec;
      const auto& at = sp.attributes;
      s += std::string(sp.kind == v::TypeConversionKind::Convert ? "|cvt." : "|bit.") + sp.target.toString() + "." +
           sp.element.toString() + "." + std::to_string(int(at.rounding)) + (at.saturate ? "s" : "") +
           (at.flush_to_zero ? "f" : "") + (at.saturate_finite ? "S" : "") + "(" + ident(c->source, depth + 1) + ")";
    }
    if (const auto& p = x.pointer())
      s += "|ptr(" + p->identity + "," + p->addressing + "," +
           (p->owner_cta ? std::to_string(*p->owner_cta) : "-") + "," +
           (p->owner_thread ? std::to_string(*p->owner_thread) : "-") + ")";
    if (!x.indices().empty()) {
      s += "|sel[";
      for (const auto& i : x.indices()) s += ident(i, depth + 1) + ";";
      s += "]";
    }
    return s;
  }

  // Refuses an operation of x (operands xs) where an operand whose value the launch does not fix
  // reaches x by a dead edge: a constant (a literal, or a value the launch fixes) decides x without
  // it. Such an edge would count as a path (a role) and a dependency (a leaf) that the value does
  // not have: a*b + 0*ex2(a) is not a path from a through ex2. Decided exactly: a product (mul,
  // mad, an fma's product) with a zero factor; and with zero; or with all ones; a logical shift by
  // at least the width; bfe of length zero; a select whose condition is fixed (a fixed condition
  // is normally resolved to its chosen operand before this step). A use of part of an operand's
  // bits (a partial mask, a lop3 table, a prmt selector, copysign) is refused by partialOperands.
  // The constant value y the launch fixes, or none: the scalar specialization's facts (integers,
  // bits, predicates) or a floating literal or specialized parameter (exact, by its literal).
  std::optional<equivalence::ConstantValue> fixed(const v::Value& y) {
    if (auto k = spec_.known(y)) return k;
    const auto x = resolve(y);
    if (x.conversion() || !x.indices().empty() || x.threadSelector()) return std::nullopt;
    const auto& type = g_.type(x);
    if (!type.isScalar() || type.element().kind() != v::ElementKind::Floating) return std::nullopt;
    const auto& op = g_.definingNode(x).operation();
    std::optional<std::string> literal;
    if (const auto* c = dynamic_cast<const v::ConstantOp*>(&op)) literal = c->literal();
    else if (const auto* p = dynamic_cast<const v::InputParameterOp*>(&op)) literal = p->specializedLiteral();
    if (!literal) return std::nullopt;
    return equivalence::constantValue(type.element().toString(), *literal);
  }

  void deadOperands(const v::Value& x, const v::Operation& op, const std::vector<v::Value>& xs) {
    const auto known = [&](std::size_t i) { return i < xs.size() ? fixed(xs[i]) : std::nullopt; };
    const auto live = [&](std::size_t i) { return i < xs.size() && !fixed(xs[i]); };
    const auto zero = [&](std::size_t i) {
      const auto k = known(i);
      return k && (k->literal == "0" || k->literal == "-0");
    };
    const auto dead = [&](std::size_t i, const char* why) {
      refuse("dead_operand", std::string("an operand reaches the value by a dead edge: ") + why,
             {{"operation", std::string(op.name())}, {"operand", i}, {"node", x.nodeId()}, {"instance", x.instanceIndex()}});
    };
    // A product's factors: k kills the other factor j.
    const auto product = [&](std::size_t f0, std::size_t f1) {
      if (zero(f0) && live(f1)) dead(f1, "a product with a zero factor");
      if (zero(f1) && live(f0)) dead(f0, "a product with a zero factor");
    };
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op); b && xs.size() == 2) {
      using B = v::BinaryOpcode;
      const auto& type = g_.type(x).element();
      const auto width = type.kind() == v::ElementKind::Floating ? 0u : type.width();
      const auto ones = [&](std::size_t i) {
        const auto k = known(i);
        return k && width && width <= 64 &&
               (k->literal == "-1" || k->literal == std::to_string(width == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << width) - 1));
      };
      switch (b->opcode()) {
        case B::Multiply: product(0, 1); break;
        case B::BitAnd:
          if (zero(0) && live(1)) dead(1, "and with zero");
          if (zero(1) && live(0)) dead(0, "and with zero");
          break;
        case B::BitOr:
          if (ones(0) && live(1)) dead(1, "or with all ones");
          if (ones(1) && live(0)) dead(0, "or with all ones");
          break;
        case B::ShiftLeft:
        case B::ShiftRight:
          if (const auto k = known(1); k && width && live(0) &&
              (b->opcode() == B::ShiftLeft || type.kind() != v::ElementKind::SignedInteger)) {
            std::uint64_t amount = 0;
            try { amount = std::stoull(k->literal); } catch (const std::exception&) { break; }
            if (k->literal[0] != '-' && amount >= width) dead(0, "a logical shift by at least the width");
          }
          break;
        default: break;
      }
    } else if (dynamic_cast<const v::FmaOp*>(&op)) {
      product(0, 1);
    } else if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      if (p->opcode() == O::MultiplyAdd || p->opcode() == O::MultiplyAddCarry) product(0, 1);
      if (p->opcode() == O::BitExtract && xs.size() == 3 && live(0)) {
        if (const auto k = known(2)) {
          std::uint64_t len = 1;
          try { len = std::stoull(k->literal); } catch (const std::exception&) {}
          if (k->literal[0] != '-' && (len & 0xff) == 0) dead(0, "bfe of length zero");
        }
      }
    } else if (dynamic_cast<const v::SelectOp*>(&op) && xs.size() == 3 && known(0)) {
      const std::size_t unchosen = known(0)->literal == "1" ? 2 : 1;
      if (live(unchosen)) dead(unchosen, "a select whose condition is fixed");
    }
  }

  // ---- Partial operands (D263): a use of only some of an operand's bits. ----
  // Per value, its bits as far as the structure decides them: `vary`, a superset of the bits that
  // may differ between runs (depend on the launch's data); every other bit is fixed, with the value
  // in `one`. `width` is 0 for a value the analysis does not track per bit (not a scalar of at most
  // 64 bits): all of its bits vary. Rules: constants and launch-fixed values are fixed; bit
  // operations with fixed controls, shifts by fixed amounts, integer extensions and truncations,
  // same-width bitcasts and bfe/bfi/prmt/lop3/shf with fixed controls are followed; any other
  // operation's result varies in every bit (its value is the op's, not decided per bit here).
  struct Bits {
    std::uint64_t vary = 0, one = 0;
    unsigned width = 0;
  };
  static std::uint64_t ones(unsigned w) { return w >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << w) - 1; }
  static Bits all(unsigned w) { return {ones(w), 0, w}; }
  // The scalar bit width of a value the analysis tracks, else 0.
  unsigned bitWidth(const v::Value& x) {
    const auto& t = g_.type(x);
    if (!t.isScalar() || t.element().isParameterized()) return 0;
    const auto w = t.element().width();
    return w <= 64 ? w : 0;
  }
  // A fixed integer value of width w, or none (a literal the specialization gives is decimal).
  std::optional<std::uint64_t> fixedBits(const v::Value& y, unsigned w) {
    const auto k = fixed(y);
    if (!k || !w) return std::nullopt;
    const auto& s = k->literal;
    if (s == "true") return 1;
    if (s == "false") return 0;
    try {
      std::size_t used = 0;
      std::uint64_t u;
      if (!s.empty() && s[0] == '-') u = std::uint64_t(std::stoll(s, &used));
      else u = std::stoull(s, &used, 0);
      if (used != s.size()) return std::nullopt;
      return u & ones(w);
    } catch (const std::exception&) {
      return std::nullopt;
    }
  }
  // An integer extension or truncation (or same-width reinterpretation) of s from `from` to `to`
  // bits; `sign`: the source is signed (sign extension).
  static Bits resize(const Bits& s, unsigned to, bool sign) {
    const auto from = s.width;
    if (!from || !to) return all(to);
    Bits r{s.vary & ones(to), s.one & ones(to), to};
    if (to > from && sign) {
      const auto high = ones(to) & ~ones(from), bit = std::uint64_t{1} << (from - 1);
      if (s.vary & bit) r.vary |= high;
      else if (s.one & bit) r.one |= high;
    }
    return r;
  }
  // A per-bit (bitwise) operation of n <= 3 same-width operands, `table` bit (b0 | b1 << 1 | b2 << 2)
  // its value: the result's bits and, per operand, the bits the result depends on for EVERY value
  // of the other operands' varying bits at that position (for all, so a mask that is itself data
  // never counts as full use).
  struct Bitwise {
    Bits result;
    std::vector<std::uint64_t> used;
  };
  static Bitwise bitwise(const std::vector<Bits>& in, unsigned table, unsigned w) {
    Bitwise out{{0, 0, w}, std::vector<std::uint64_t>(in.size(), 0)};
    const auto n = in.size();
    for (unsigned i = 0; i < w; ++i) {
      const auto bit = std::uint64_t{1} << i;
      // The possible combinations at bit i: a varying operand takes both values.
      std::vector<unsigned> combos;
      for (unsigned c = 0; c < (1u << n); ++c) {
        bool ok = true;
        for (std::size_t k = 0; k < n; ++k)
          if (!(in[k].vary & bit) && bool(c >> k & 1) != bool(in[k].one & bit)) ok = false;
        if (ok) combos.push_back(c);
      }
      bool v0 = false, v1 = false;
      for (const auto c : combos) (table >> c & 1 ? v1 : v0) = true;
      if (v0 && v1) out.result.vary |= bit;
      else if (v1) out.result.one |= bit;
      for (std::size_t k = 0; k < n; ++k) {
        bool every = true;
        for (const auto c : combos)
          every &= (table >> c & 1) != (table >> (c ^ (1u << k)) & 1);
        if (every) out.used[k] |= bit;
      }
    }
    return out;
  }

  // The bits of value x, memoized by identity.
  Bits bits(v::Value x, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    const auto key = ident(x);
    if (auto it = bits_.find(key); it != bits_.end()) return it->second;
    const auto r = bitsStep(x, depth);
    bits_.emplace(key, r);
    return r;
  }
  Bits bitsStep(const v::Value& x, unsigned depth) {
    const auto w = bitWidth(x);
    if (!w) return all(0);
    const bool integer = g_.type(x).element().isBitLike();
    if (integer)
      if (const auto k = fixedBits(x, w)) return {0, *k, w};
    if (!integer) return all(w);
    if (!x.indices().empty()) return all(w);
    if (const auto& c = x.conversion()) {
      const auto& src = c->source;
      const auto sw = bitWidth(src);
      const auto& se = g_.type(src).element();
      if (!sw || !se.isBitLike()) return all(w);
      if (c->spec->kind == v::TypeConversionKind::Bitcast) return sw == w ? bits(src, depth + 1) : all(w);
      if (c->spec->attributes.saturate) return all(w);
      return resize(bits(src, depth + 1), w, se.kind() == v::ElementKind::SignedInteger);
    }
    const auto& op = g_.definingNode(x).operation();
    const auto xs = g_.instanceOperands(x).toVector();
    const auto in = [&](std::size_t i) { return bits(xs[i], depth + 1); };
    if (dynamic_cast<const v::BitcastOp*>(&op) && xs.size() == 1)
      return bitWidth(xs[0]) == w && g_.type(xs[0]).element().isBitLike() ? in(0) : all(w);
    if (const auto* cv = dynamic_cast<const v::ConvertOp*>(&op); cv && xs.size() == 1) {
      const auto& se = g_.type(xs[0]).element();
      if (!bitWidth(xs[0]) || !se.isBitLike() || cv->attributes().saturate) return all(w);
      return resize(in(0), w, se.kind() == v::ElementKind::SignedInteger);
    }
    if (const auto* u = dynamic_cast<const v::UnaryOp*>(&op); u && u->opcode() == v::UnaryOpcode::BitNot && xs.size() == 1)
      return bitwise({in(0)}, 0b01, w).result;
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op); b && xs.size() == 2) {
      using B = v::BinaryOpcode;
      switch (b->opcode()) {
        case B::BitAnd: return bitwise({in(0), in(1)}, 0b1000, w).result;
        case B::BitOr: return bitwise({in(0), in(1)}, 0b1110, w).result;
        case B::BitXor: return bitwise({in(0), in(1)}, 0b0110, w).result;
        case B::ShiftLeft:
        case B::ShiftRight: {
          const auto k = fixedBits(xs[1], bitWidth(xs[1]));
          if (!k) return all(w);
          return shifted(in(0), *k, b->opcode() == B::ShiftLeft, g_.type(x).element().kind() == v::ElementKind::SignedInteger).first;
        }
        default: return all(w);
      }
    }
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      if (const auto r = ptxBits(*p, xs, w, depth)) return r->first;
    }
    return all(w);
  }
  // A shift of s by fixed amount k: the result's bits and the operand bits it reads.
  static std::pair<Bits, std::uint64_t> shifted(const Bits& s, std::uint64_t k, bool left, bool sign) {
    const auto w = s.width;
    if (!w) return {all(0), 0};
    Bits r{0, 0, w};
    std::uint64_t used = 0;
    for (unsigned i = 0; i < w; ++i) {
      std::optional<unsigned> from;  // the operand bit result bit i copies
      if (left) { if (k <= i) from = unsigned(i - k); }
      else if (k < w && i + k < w) from = unsigned(i + k);
      else if (sign) from = w - 1;
      if (!from) continue;
      const auto f = std::uint64_t{1} << *from, t = std::uint64_t{1} << i;
      used |= f;
      if (s.vary & f) r.vary |= t;
      else if (s.one & f) r.one |= t;
    }
    return {r, used};
  }
  // A routing of operand bits to result bits (each result bit a copy of one operand bit, or a
  // constant): result bit i is `src[i]` = (operand, bit) or none (zero).
  struct Route {
    std::vector<std::optional<std::pair<std::size_t, unsigned>>> src;
  };
  Bits routed(const Route& rt, const std::vector<Bits>& in, unsigned w, std::vector<std::uint64_t>& used) {
    Bits r{0, 0, w};
    used.assign(in.size(), 0);
    for (unsigned i = 0; i < w && i < rt.src.size(); ++i) {
      if (!rt.src[i]) continue;
      const auto [k, j] = *rt.src[i];
      const auto f = std::uint64_t{1} << j, t = std::uint64_t{1} << i;
      used[k] |= f;
      if (in[k].vary & f) r.vary |= t;
      else if (in[k].one & f) r.one |= t;
    }
    return r;
  }
  // bfe, bfi, prmt (default mode), lop3, shf with fixed controls: the result's bits and, per
  // operand, the bits it reads (controls: none; they must be fixed). None when a control is not
  // fixed or the form is not one of these.
  std::optional<std::pair<Bits, std::vector<std::uint64_t>>> ptxBits(const v::PtxScalarOp& p, const std::vector<v::Value>& xs,
                                                                      unsigned w, unsigned depth) {
    using O = v::PtxScalarOpcode;
    const auto control = [&](std::size_t i) { return i < xs.size() ? fixedBits(xs[i], bitWidth(xs[i])) : std::nullopt; };
    const auto in = [&](std::size_t i) { return bits(xs[i], depth + 1); };
    for (const auto& m : p.attributes().modifiers)
      if (m != "b32" && m != "u32" && m != "s32" && m != "b64" && m != "u64" && m != "s64" &&
          !(p.opcode() == O::FunnelShift && (m == "l" || m == "r" || m == "clamp" || m == "wrap")))
        return std::nullopt;  // a mode (prmt.f4e, ...) this analysis does not model
    const bool sign = p.sourceType().element().kind() == v::ElementKind::SignedInteger;
    std::vector<std::uint64_t> used(xs.size(), 0);
    Route rt;
    rt.src.assign(w, std::nullopt);
    std::vector<Bits> ins;
    switch (p.opcode()) {
      case O::BitExtract: {  // bfe a, pos, len: d[i] = i < len && pos + i <= msb ? a[pos + i] : sbit
        const auto pos = control(1), len = control(2);
        if (xs.size() != 3 || !pos || !len) return std::nullopt;
        const auto ps = unsigned(*pos & 0xff), ln = unsigned(*len & 0xff), msb = w - 1;
        std::optional<std::pair<std::size_t, unsigned>> sbit;
        if (sign && ln) sbit = std::pair{std::size_t{0}, std::min(ps + ln - 1, msb)};
        for (unsigned i = 0; i < w; ++i) rt.src[i] = i < ln && ps + i <= msb ? std::optional(std::pair{std::size_t{0}, ps + i}) : sbit;
        ins = {in(0), Bits{0, 0, 32}, Bits{0, 0, 32}};
        break;
      }
      case O::BitInsert: {  // bfi f, b, pos, len: d = b; d[pos + i] = f[i] for i < len, pos + i <= msb
        const auto pos = control(2), len = control(3);
        if (xs.size() != 4 || !pos || !len) return std::nullopt;
        const auto ps = unsigned(*pos & 0xff), ln = unsigned(*len & 0xff);
        for (unsigned i = 0; i < w; ++i)
          rt.src[i] = i >= ps && i - ps < ln ? std::pair{std::size_t{0}, i - ps} : std::pair{std::size_t{1}, i};
        ins = {in(0), in(1), Bits{0, 0, 32}, Bits{0, 0, 32}};
        break;
      }
      case O::BytePermute: {  // prmt a, b, c: byte k of d is byte c[4k+2..4k] of {b, a}, or its sign
        const auto c = control(2);
        if (xs.size() != 3 || !c || w != 32) return std::nullopt;
        for (unsigned k = 0; k < 4; ++k) {
          const auto s = unsigned(*c >> (4 * k) & 0xf), byte = s & 7;
          const std::size_t from = byte < 4 ? 0 : 1;
          const unsigned base = (byte & 3) * 8;
          for (unsigned j = 0; j < 8; ++j)
            rt.src[8 * k + j] = std::pair{from, s & 8 ? base + 7 : base + j};
        }
        ins = {in(0), in(1), Bits{0, 0, 32}};
        break;
      }
      case O::FunnelShift: {  // shf.{l,r}.{clamp,wrap} a, b, n over the 64-bit {b, a}
        const auto n = control(2);
        bool left = false, wrap = false;
        for (const auto& m : p.attributes().modifiers) left |= m == "l", wrap |= m == "wrap";
        if (xs.size() != 3 || !n || w != 32) return std::nullopt;
        const auto k = unsigned(wrap ? *n & 31 : std::min<std::uint64_t>(*n, 32));
        for (unsigned i = 0; i < 32; ++i) {
          const int at = left ? int(i) + 32 - int(k) : int(i + k);  // a bit of {b, a}
          if (at >= 0 && at < 64) rt.src[i] = std::pair{std::size_t(at < 32 ? 0 : 1), unsigned(at % 32)};
        }
        ins = {in(0), in(1), Bits{0, 0, 32}};
        break;
      }
      case O::Logic3: {
        const auto t = control(3);
        if (xs.size() != 4 || !t) return std::nullopt;
        // lop3's table: bit (a << 2 | b << 1 | c); bitwise() indexes b0 | b1 << 1 | b2 << 2.
        unsigned table = 0;
        for (unsigned c = 0; c < 8; ++c)
          if (*t >> ((c & 1) << 2 | (c >> 1 & 1) << 1 | (c >> 2 & 1)) & 1) table |= 1u << c;
        auto bw = bitwise({in(0), in(1), in(2)}, table, w);
        bw.used.push_back(0);
        return std::pair{bw.result, bw.used};
      }
      default:
        return std::nullopt;
    }
    const auto r = routed(rt, ins, w, used);
    return std::pair{r, used};
  }

  // Whether `used` includes every bit operand y may vary in (an untracked operand or an undecided
  // use: no), or y is a float's own bits and the bits not used are its low mantissa bits (a
  // rounding toward zero to a narrower float: sign, exponent and leading mantissa bits kept,
  // `__float_as_uint(v) >> 16` storing a bf16, numerics inside the atom like cvt.rz.bf16.f32).
  bool covers(const v::Value& y, std::optional<std::uint64_t> used) {
    const auto ow = bitWidth(y);
    if (!used || !ow) return false;
    const auto unused = bits(y).vary & ~*used & ones(ow);
    if (!unused) return true;
    const auto e = floatExponent(y);
    const auto low = 64u - unsigned(__builtin_clzll(unused));  // bits [0, low) hold every unused bit
    return e && unused == ones(low) && (*used & ones(ow) & ~ones(low)) == (ones(ow) & ~ones(low)) && ow - low >= 1 + *e;
  }
  // The exponent width of the float whose bits y is (a chain of same-width bitcasts of an f16,
  // bf16, f32 or f64 value), or none.
  std::optional<unsigned> floatExponent(v::Value y) {
    const auto w = bitWidth(y);
    for (unsigned n = 0; n < 16; ++n) {
      y = resolve(y);
      if (!y.indices().empty() || bitWidth(y) != w) return std::nullopt;
      if (g_.type(y).element().kind() == v::ElementKind::Floating) break;
      if (const auto& c = y.conversion()) {
        if (c->spec->kind != v::TypeConversionKind::Bitcast) return std::nullopt;
        y = c->source;
      } else if (dynamic_cast<const v::BitcastOp*>(&g_.definingNode(y).operation())) {
        y = g_.instanceOperands(y).toVector().at(0);
      } else {
        return std::nullopt;
      }
    }
    if (g_.type(y).element().kind() != v::ElementKind::Floating || bitWidth(y) != w) return std::nullopt;
    const auto name = g_.type(y).element().toString();
    if (name == "f16") return 5;
    if (name == "bf16" || name == "f32") return 8;
    if (name == "f64") return 11;
    return std::nullopt;
  }
  // The partial uses found (the first 8, for refusal details).
  void note(const v::Value& x, const std::string& what, std::size_t i, const v::Value& y, std::optional<std::uint64_t> used,
            const char* why) {
    Json d{{"operation", what}, {"operand", i}, {"node", x.nodeId()}, {"instance", x.instanceIndex()}, {"why", why}};
    if (const auto ow = bitWidth(y)) d["operand_bits"] = ow, d["varying"] = bits(y).vary, d["used"] = used ? Json(*used) : Json(nullptr);
    if (partial_.size() < 8 && std::find(partial_.begin(), partial_.end(), d) == partial_.end()) partial_.push_back(d);
  }
  // Which operands of an operation of x (operands xs) it uses in full: every bit the operand may
  // vary in (bits()) reaches the result. A partial use (reload(p) & 0xffff depends on p's low half
  // only) is not a full path and does not make its leaves full dependencies (partial_operand).
  // Decided exactly: bitwise operations (per bit, for every value of the other operands' varying
  // bits), shifts, bfe, bfi, prmt (default mode), lop3, shf with fixed controls; integer
  // truncation; integer products, quotients and remainders with a fixed operand; float abs (no
  // sign) and copysign (a sign, a magnitude); slct's integer selector (its sign); a component of an
  // operation's matrix result. Any of these with a control the launch does not fix (a data-dependent
  // mask, shift, selector or table) is partial (undecided). Full (every bit can reach the result):
  // other float operations, integer add, sub, min, max, not, compares, selects' values and
  // conditions. An operand that is fixed or reaches no leaf is full (nothing to credit).
  std::vector<char> operandsFull(const v::Value& x, const v::Operation& op, const std::vector<v::Value>& xs) {
    std::vector<char> full(xs.size(), 1);
    const auto w = bitWidth(x);
    const auto live = [&](std::size_t i) { return i < xs.size() && !fixed(xs[i]) && !leaves(xs[i]).empty(); };
    const auto check = [&](std::size_t i, std::optional<std::uint64_t> used, const char* why) {
      if (live(i) && !covers(xs[i], used)) full[i] = 0, note(x, std::string(op.name()), i, xs[i], used, why);
    };
    if (!x.indices().empty()) {  // a component of the operation's matrix result
      for (std::size_t i = 0; i < xs.size(); ++i) check(i, std::nullopt, "a component of an operation's matrix result");
      return full;
    }
    const auto all_used = [&](std::size_t i) -> std::optional<std::uint64_t> { return ones(bitWidth(xs[i])); };
    const auto& elem = g_.type(x).element();
    const bool integer = elem.isInteger();
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op); b && xs.size() == 2) {
      using B = v::BinaryOpcode;
      std::vector<Bits> in{bits(xs[0]), bits(xs[1])};
      switch (b->opcode()) {
        case B::BitAnd: case B::BitOr: case B::BitXor: {
          if (!w) { check(0, std::nullopt, "a bit operation on an untracked value"); check(1, std::nullopt, "a bit operation on an untracked value"); return full; }
          const unsigned table = b->opcode() == B::BitAnd ? 0b1000 : b->opcode() == B::BitOr ? 0b1110 : 0b0110;
          const auto bw = bitwise(in, table, w);
          check(0, bw.used[0], "a bitwise operation reads only some bits");
          check(1, bw.used[1], "a bitwise operation reads only some bits");
          return full;
        }
        case B::ShiftLeft: case B::ShiftRight: {
          const auto k = fixedBits(xs[1], bitWidth(xs[1]));
          check(1, std::nullopt, "a data-dependent shift amount");
          if (!k || !w) { check(0, std::nullopt, "a shift by an amount the launch does not fix"); return full; }
          const bool sign = elem.kind() == v::ElementKind::SignedInteger;
          check(0, shifted(in[0], *k, b->opcode() == B::ShiftLeft, sign).second, "a shift reads only some bits");
          return full;
        }
        case B::Multiply:
          if (!integer) return full;
          for (std::size_t i = 0; i < 2; ++i) {
            const auto mode = b->attributes().multiply_mode;
            check(i, productUses(bitWidth(xs[i]), in[1 - i], mode == v::IntegerMultiplyMode::Wide, mode == v::IntegerMultiplyMode::High),
                  "an integer product whose other factor's trailing zeros are not decided zero (it reads only some bits, or which is not decided)");
          }
          return full;
        case B::Divide: case B::Remainder:
          if (!integer) return full;
          for (std::size_t i = 0; i < 2; ++i) {
            const auto c = fixedBits(xs[1 - i], bitWidth(xs[1 - i]));
            const bool exact = c && b->opcode() == B::Divide && i == 0 && *c == 1;
            check(i, exact ? all_used(i) : std::nullopt, "an integer quotient or remainder (bits read not decided)");
          }
          return full;
        default:
          return full;
      }
    }
    if (const auto* u = dynamic_cast<const v::UnaryOp*>(&op); u && xs.size() == 1) {
      if (u->opcode() == v::UnaryOpcode::Absolute && elem.kind() == v::ElementKind::Floating && w)
        check(0, ones(w - 1), "float abs drops the sign bit");
      return full;
    }
    if (const auto* cv = dynamic_cast<const v::ConvertOp*>(&op); cv && xs.size() == 1) {
      convertOperand(x, xs[0], cv->attributes().saturate, check);
      return full;
    }
    if (dynamic_cast<const v::BitcastOp*>(&op) && xs.size() == 1) {
      if (bitWidth(xs[0]) != w || !w) check(0, std::nullopt, "a bitcast between widths");
      return full;
    }
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      switch (p->opcode()) {
        case O::BitExtract: case O::BitInsert: case O::BytePermute: case O::Logic3: case O::FunnelShift: {
          const auto r = ptxBits(*p, xs, w, 0);
          const std::size_t data = p->opcode() == O::BitExtract ? 1 : p->opcode() == O::Logic3 ? 3 : 2;
          for (std::size_t i = 0; i < xs.size(); ++i) {
            if (i >= data) check(i, std::nullopt, "a data-dependent control (position, length, selector, table or amount)");
            else check(i, r ? std::optional(r->second[i]) : std::nullopt, "a bit-field operation reads only some bits");
          }
          return full;
        }
        case O::CopySign: {  // copysign a, b: b's magnitude with a's sign
          const auto ow = xs.empty() ? 0u : bitWidth(xs[0]);
          if (!ow || xs.size() != 2) { check(0, std::nullopt, "copysign"); check(1, std::nullopt, "copysign"); return full; }
          check(0, std::uint64_t{1} << (ow - 1), "copysign reads only the sign of its first operand");
          check(1, ones(ow - 1), "copysign reads all but the sign of its second operand");
          return full;
        }
        case O::SelectSign:
          if (xs.size() == 3 && g_.type(xs[2]).element().isInteger() && bitWidth(xs[2]))
            check(2, std::uint64_t{1} << (bitWidth(xs[2]) - 1), "slct reads only the sign of an integer selector");
          return full;
        case O::MultiplyAdd: case O::MultiplyAddCarry: {
          if (!integer || xs.size() != 3) return full;
          bool wide = false, high = false;
          for (const auto& m : p->attributes().modifiers) wide |= m == "wide", high |= m == "hi";
          for (std::size_t i = 0; i < 2; ++i)
            check(i, productUses(bitWidth(xs[i]), bits(xs[1 - i]), wide, high),
                  "an integer multiply-add whose other factor's trailing zeros are not decided zero (it reads only some bits, or which is not decided)");
          return full;
        }
        default:
          return full;
      }
    }
    return full;
  }
  // The bits of a factor (width ow) an integer product reads, from the other factor's bits o, or
  // none (undecided). Low half (mul.lo, mad.lo): flipping factor bit i changes the product by
  // o * 2^i, which vanishes mod 2^w exactly when o has at least w - i trailing zeros; so with t the
  // other factor's trailing-zero count the product reads the low ow - t bits, and t is decided only
  // when o's low bits are fixed zero up to a fixed one (a low bit that may vary, as in x * (x << 1),
  // leaves t undecided). Wide: every bit unless o is zero; decided when some bit of o is fixed one.
  // High half: not decided. The analysis does not relate the two factors (x * (x << 1)): o is a
  // superset of the other factor's varying bits, so a fixed bit is fixed for every value of x.
  static std::optional<std::uint64_t> productUses(unsigned ow, const Bits& o, bool wide, bool high) {
    if (high || !ow) return std::nullopt;
    if (wide) return o.one & ~o.vary & ones(o.width) ? std::optional(ones(ow)) : std::nullopt;
    for (unsigned t = 0; t < o.width; ++t) {
      const auto b = std::uint64_t{1} << t;
      if (o.vary & b) return std::nullopt;
      if (o.one & b) return t >= ow ? 0 : ones(ow - t);
    }
    return std::nullopt;  // o is zero (a dead product) or untracked
  }
  // A conversion of `source` to x's type: an integer truncation reads the low bits only.
  template <class Check>
  void convertOperand(const v::Value& x, const v::Value& source, bool saturate, const Check& check) {
    const auto w = bitWidth(x), sw = bitWidth(source);
    const auto& se = g_.type(source).element();
    const auto& te = g_.type(x).element();
    if (!se.isBitLike() || !te.isBitLike() || saturate) return;  // float conversions, saturation: every bit
    if (!w || !sw) check(0, std::nullopt, "an integer conversion of an untracked value");
    else if (w < sw) check(0, ones(w), "an integer truncation reads only the low bits");
  }
  // Whether the conversion step of x (a value with a conversion) uses all of its source: not a
  // truncation, a bitcast between widths, or a component of a reinterpreted value.
  bool conversionFull(const v::Value& x) {
    const auto& c = *x.conversion();
    const auto& src = c.source;
    bool full = true;
    const auto check = [&](std::size_t, std::optional<std::uint64_t> used, const char* why) {
      if (!fixed(src) && !leaves(src).empty() && !covers(src, used))
        full = false, note(x, c.spec->kind == v::TypeConversionKind::Convert ? "convert" : "bitcast", 0, src, used, why);
    };
    if (!x.indices().empty()) {
      check(0, std::nullopt, "a component of a reinterpreted value");
      return full;
    }
    const auto conv = x.withoutSelection();
    if (c.spec->kind == v::TypeConversionKind::Bitcast) {
      if (g_.type(conv).bitSize() != g_.type(src).bitSize()) check(0, std::nullopt, "a bitcast between widths");
      return full;
    }
    convertOperand(conv, src, c.spec->attributes.saturate, check);
    return full;
  }

  // One step of the complete dataflow of a resolved value x (no thread selector): its successors
  // (index values first, then the conversion source, the replaced load's stored value or the
  // operation's operands), whether x is a leaf (an initial-memory load, a constant, a parameter or
  // a coordinate) and whether x's operation is a transcendental. Anything this cannot decide is
  // unsupported: a load at a data-dependent position or without a read resolution, a read
  // reassembled from stored values, and an operation of a kind not listed here.
  struct Flow {
    std::vector<v::Value> next;
    std::vector<char> full;   // per next: the edge uses all of that value's bits (operandsFull)
    std::size_t indices = 0;  // next[0, indices) are x's index values
    bool leaf = false, trans = false, load = false;
  };
  Flow flow(const v::Value& x) {
    Flow f;
    for (const auto& i : x.indices()) f.next.push_back(i), f.full.push_back(1);
    f.indices = f.next.size();
    if (const auto& c = x.conversion()) {
      f.next.push_back(c->source);
      f.full.push_back(conversionFull(x));
      return f;
    }
    const auto& op = g_.definingNode(x).operation();
    if (dynamic_cast<const v::LoadOp*>(&op)) {
      f.load = true;
      const MemoryInstance key{x.nodeId(), x.instanceIndex()};
      if (m_.symbolic_initial_reads.contains(key)) return f.leaf = true, f;
      if (m_.symbolic_read_positions.contains(key))
        unsupported("symbolic_read_position", "a read at a data-dependent position (needs table premises)");
      if (const auto* r = m_.readReplacement(key)) {
        if (const auto* value = std::get_if<v::Value>(r)) {
          f.next.push_back(x.indices().empty() ? *value : g_.component(*value, {x.indices().begin(), x.indices().end()}));
          f.full.push_back(1);
          return f;
        }
        if (std::get_if<InitialMemorySymbol>(r)) return f.leaf = true, f;
        const auto initial = [](const ReinterpretedRead& p) { return std::holds_alternative<InitialMemorySymbol>(p.source); };
        bool all = true;
        if (const auto* p = std::get_if<ReinterpretedRead>(r)) all = initial(*p);
        else for (const auto& p : std::get<AssembledRead>(*r).pieces) all &= initial(p);
        if (!all) unsupported("reassembled_read", "a read reassembled from stored values (its dataflow per byte is not resolved)");
        return f.leaf = true, f;
      }
      if (m_.initialRead(key)) return f.leaf = true, f;
      unsupported("unresolved_load", "a load with no read resolution", {{"node", key.node}, {"instance", key.instance}});
    }
    if (dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op) ||
        dynamic_cast<const v::CoordinateOp*>(&op))
      return f.leaf = true, f;
    const bool known = dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::BinaryOp*>(&op) ||
                       dynamic_cast<const v::FmaOp*>(&op) || dynamic_cast<const v::IdealExp2Op*>(&op) ||
                       dynamic_cast<const v::ConvertOp*>(&op) || dynamic_cast<const v::BitcastOp*>(&op) ||
                       dynamic_cast<const v::CompareOp*>(&op) || dynamic_cast<const v::SelectOp*>(&op) ||
                       dynamic_cast<const v::ThreadForwardOp*>(&op) || dynamic_cast<const v::PtxScalarOp*>(&op);
    if (!known)
      unsupported("operation_kind", "an operation kind the structural check does not classify",
                  {{"operation", std::string(op.name())}});
    f.trans = transcendental(op);
    const auto operands = g_.instanceOperands(x).toVector();
    deadOperands(x, op, operands);
    const auto full = operandsFull(x, op, operands);
    for (std::size_t i = 0; i < operands.size(); ++i) f.next.push_back(operands[i]), f.full.push_back(full[i]);
    return f;
  }

  // The leaf set of value x, cut values (stop_, by identity) counting as leaves. Memoized by identity.
  Set leaves(v::Value x, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    const auto key = ident(x);
    if (auto it = stop_.find(key); it != stop_.end()) return {{it->second, it->second + 1}};
    if (auto it = memo_.find(key); it != memo_.end()) return *it->second;
    Set s;
    if (x.conversion()) {
      s = leaves(x.conversion()->source, depth + 1);
      for (const auto& i : x.indices()) s = unite(s, leaves(i, depth + 1));
    } else {
      const auto& op = g_.definingNode(x).operation();
      if (dynamic_cast<const v::LoadOp*>(&op)) {
        s = *loadLeaves(x, false);
      } else if (dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op) ||
                 dynamic_cast<const v::CoordinateOp*>(&op)) {
      } else if (dynamic_cast<const v::OpaqueValueOp*>(&op) || dynamic_cast<const v::FailedClcResponseOp*>(&op)) {
        unsupported("undetermined_value", "a value whose bits the IR does not determine");
      } else {
        const auto operands = g_.instanceOperands(x).toVector();
        deadOperands(x, op, operands);
        for (const auto& a : operands) s = unite(s, leaves(a, depth + 1));
        for (const auto& i : x.indices()) s = unite(s, leaves(i, depth + 1));
      }
    }
    memo_.emplace(key, std::make_shared<const Set>(s));
    return s;
  }

  // The leaves x reaches by full uses only (flow's full edges): the dependencies that hold in full.
  // leaves() over every edge is the dependencies that may hold at all; a leaf in leaves() and not
  // here is reached only through partial uses (partial_operand). Memoized by identity.
  Set fullLeaves(v::Value x, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    const auto key = ident(x);
    if (auto it = stop_.find(key); it != stop_.end()) return {{it->second, it->second + 1}};
    if (auto it = full_memo_.find(key); it != full_memo_.end()) return *it->second;
    Set s;
    const auto& op = g_.definingNode(x).operation();
    if (!x.conversion() && dynamic_cast<const v::LoadOp*>(&op)) {
      s = *loadLeaves(x, false, true);
    } else {
      leaves(x, depth);  // its refusals and unsupported steps first
      const auto f = flow(x);
      for (std::size_t i = 0; i < f.next.size(); ++i)
        if (f.full[i]) s = unite(s, fullLeaves(f.next[i], depth + 1));
    }
    full_memo_.emplace(key, std::make_shared<const Set>(s));
    return s;
  }
  // Refuses when x's leaves are not all reached by full uses.
  void requireFull(const v::Value& x, const Set& all, const Set& full, const Json& where) {
    if (full == all) return;
    auto d = where;
    d["leaves"] = render(all);
    d["full_leaves"] = render(full);
    d["partial_uses"] = partial_;
    refuse("partial_operand", "a leaf is reached only through uses of part of a value's bits (or of bits not decided)", d);
    (void)x;
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
          x.conversion()->spec->target.element().toString() == convert && x.indices().empty()) {
        x = resolve(x.conversion()->source);
      } else if (!x.conversion() && x.indices().empty() && dynamic_cast<const v::ConvertOp*>(&g_.definingNode(x).operation()) &&
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

  // The transcendentals of the role signatures: ex2, lg2, sin, cos, tanh (every PTX scalar opcode
  // is listed, so a new one is a compile warning here, never silently a non-transcendental).
  static bool transcendental(const v::Operation& op) {
    if (dynamic_cast<const v::IdealExp2Op*>(&op)) return true;
    const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op);
    if (!p) return false;
    using O = v::PtxScalarOpcode;
    switch (p->opcode()) {
      case O::Exp2: case O::Log2: case O::Sin: case O::Cos: case O::Tanh:
        return true;
      case O::AddCarry: case O::SubtractCarry: case O::MultiplyAdd: case O::MultiplyAddCarry:
      case O::SetPredicate: case O::SetValue: case O::SelectSign: case O::AddressSpaceConvert:
      case O::ClusterSharedAddress: case O::BitReverse: case O::BitFind: case O::CountLeadingZeros:
      case O::PopulationCount: case O::BitExtract: case O::BitInsert: case O::BytePermute: case O::Logic3:
      case O::FunnelShift: case O::CopySign: case O::LogicalNot: case O::TestPredicate:
      case O::Reciprocal: case O::Sqrt: case O::ReciprocalSqrt:
        return false;
    }
    unsupported("operation_kind", "a PTX scalar opcode the structural check does not classify");
  }

  // Reachability over the complete dataflow below a value: `all`, the source elements some path
  // from the value reaches; `through`, those reached by a path that passes a transcendental
  // operation strictly above the source. No cut ends a path. A source is the argument's value:
  // for an input argument an initial-memory load (its elements), for a body argument a value the
  // producer's output is bound to (by identity, its cut element). `full`, `through_full`: the same
  // over full uses only (flow's full edges). A claim that a path exists (an element reaches the
  // output, through a transcendental) needs a full path; a claim that none exists (avoids) counts
  // every path, partial ones too.
  struct Reach {
    Set all, through, full, through_full;
  };
  using ReachMemo = std::map<std::string, std::shared_ptr<const Reach>>;
  template <class Source>
  const Reach& reach(v::Value x, ReachMemo& memo, const Source& source, unsigned depth = 0) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    auto key = ident(x);
    if (auto it = memo.find(key); it != memo.end()) return *it->second;
    const auto f = flow(x);
    Reach r;
    Set below, below_full;
    for (std::size_t i = 0; i < f.next.size(); ++i) {
      const auto& s = reach(f.next[i], memo, source, depth + 1);
      below = unite(below, s.all);
      r.through = unite(r.through, s.through);
      if (f.full[i]) {
        below_full = unite(below_full, s.full);
        r.through_full = unite(r.through_full, s.through_full);
      }
    }
    if (f.trans) r.through = unite(r.through, below), r.through_full = unite(r.through_full, below_full);
    const auto here = source(x, key, f);
    r.all = unite(below, here);
    r.full = unite(below_full, here);
    return *memo.emplace(std::move(key), std::make_shared<const Reach>(std::move(r))).first->second;
  }
  // Sources of an input argument: initial-memory loads.
  const Reach& inputReach(const v::Value& root) {
    return reach(root, reach_in_, [&](const v::Value& x, const std::string&, const Flow& f) {
      return f.load && f.leaf ? *loadLeaves(x, false) : Set{};
    });
  }
  // Sources of body argument on producer m: the values m's stored output elements hold (by
  // identity; one value at two elements is ambiguous).
  const Reach& bodyReach(const v::Value& root, std::size_t m) {
    auto [src, fresh] = sources_.try_emplace(m);
    if (fresh) {
      const auto space = t_.objects.size() + m;
      const auto& values = stored_.at(*t_.nodes[m].out);
      for (std::size_t e = 0; e < values.size(); ++e) {
        if (!values[e]) continue;
        const auto id = (std::uint64_t(space) << kShift) | e;
        if (auto [it, added] = src->second.emplace(ident(*values[e]), id); !added && it->second != id)
          refuse("ambiguous_cut", "one value stored at two elements of an inner node", {{"node", m}});
      }
    }
    const auto& map = src->second;
    return reach(root, reach_body_[m], [&](const v::Value&, const std::string& key, const Flow&) {
      const auto it = map.find(key);
      return it == map.end() ? Set{} : Set{{it->second, it->second + 1}};
    });
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

  // Memoized per argument set k (roles are few, so one memo per (root, k)), by value identity.
  using Memo = std::map<std::string, SetRef>;

  // The leaves of k that x reaches in full (fullLeaves) as a factor of a multiply (or an fma's
  // product) whose other factor reaches no leaf of k, along full uses only.
  Set factorLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, const auto& full, unsigned d) {
      Set s;
      const auto* b = dynamic_cast<const v::BinaryOp*>(&op);
      const bool product = (b && b->opcode() == v::BinaryOpcode::Multiply) || dynamic_cast<const v::FmaOp*>(&op);
      if (product && xs.size() >= 2) {
        const auto l0 = meet(leaves(xs[0], d + 1), k), l1 = meet(leaves(xs[1], d + 1), k);
        if (l1.empty() && full[0]) s = unite(s, meet(fullLeaves(xs[0], d + 1), k));
        if (l0.empty() && full[1]) s = unite(s, meet(fullLeaves(xs[1], d + 1), k));
      }
      for (std::size_t i = 0; i < xs.size(); ++i)
        if (full[i]) s = unite(s, factorLeaves(xs[i], k, memo, d + 1));
      return s;
    }, [&](const v::Value& i, unsigned d) { return factorLeaves(i, k, memo, d); },
    [&](const v::Value&, unsigned) { return Set{}; });
  }

  // The leaves of k that x reaches on some path that passes no product whose other factor reaches
  // no leaf of k (a path that bypasses the multiply). An index value is never a factor. Every path
  // counts, partial ones too.
  Set bypassLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, const auto&, unsigned d) {
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
    }, [&](const v::Value& i, unsigned d) { return operandBypass(i, k, memo, d); },
    [&](const v::Value& n, unsigned d) { return bypassLeaves(n, k, memo, d); });
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
  // bit operations with constant operands, or as both operands of one add. A partial use is not
  // an addend path (its leaves count here).
  Set nonAddendLeaves(v::Value x, const Set& k, Memo& memo, unsigned depth = 0) {
    return walk(x, memo, depth, [&](const v::Operation& op, const auto& xs, const auto& full, unsigned d) {
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
        // A bit operation whose other operands are constants (no leaves) carries its one operand,
        // when it uses all of that operand's bits.
        std::vector<std::size_t> live;
        for (std::size_t i = 0; i < xs.size(); ++i)
          if (!leaves(xs[i], d + 1).empty()) live.push_back(i);
        const bool bits = code == B::ShiftLeft || code == B::ShiftRight || code == B::BitAnd || code == B::BitOr;
        if (bits && live.size() == 1 && full[live[0]]) return nonAddendLeaves(xs[live[0]], k, memo, d + 1);
        for (const auto& a : xs) s = unite(s, meet(leaves(a, d + 1), k));
      }
      return s;
    }, [&](const v::Value& i, unsigned d) { return meet(leaves(i, d), k); },
    [&](const v::Value& n, unsigned d) { return meet(leaves(n, d), k); });
  }

  // A memoized walk of x's complete dataflow (flow): through conversions and replaced loads (a
  // partial conversion: `partial` instead), `f` on any other operation with its operands and their
  // full flags, `index` on each index value; a leaf gives the empty set. Undecided steps are
  // unsupported (flow).
  template <class F, class I, class P>
  Set walk(v::Value x, Memo& memo, unsigned depth, const F& f, const I& index, const P& partial) {
    if (depth > 20000) unsupported("depth", "value DAG deeper than 20000");
    x = resolve(x);
    auto key = ident(x);
    if (auto it = memo.find(key); it != memo.end()) return *it->second;
    const auto fl = flow(x);
    Set s;
    for (std::size_t i = 0; i < fl.indices; ++i) s = unite(s, index(fl.next[i], depth + 1));
    const std::vector<v::Value> rest(fl.next.begin() + std::ptrdiff_t(fl.indices), fl.next.end());
    const std::vector<char> full(fl.full.begin() + std::ptrdiff_t(fl.indices), fl.full.end());
    if (x.conversion() || fl.load) {
      for (std::size_t i = 0; i < rest.size(); ++i)
        s = unite(s, full[i] ? walk(rest[i], memo, depth + 1, f, index, partial) : partial(rest[i], depth + 1));
    } else if (!fl.leaf) {
      s = unite(s, f(g_.definingNode(x).operation(), rest, full, depth));
    }
    memo.emplace(std::move(key), std::make_shared<const Set>(s));
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
  // role is a fact about the paths from the argument's value to m's output over the complete
  // dataflow (no cut ends a path): for an input argument from its initial-memory loads, for a body
  // argument from the values its producer's stored output elements hold (by identity). Every
  // element of the argument must reach the output; through: each by some path that passes a
  // transcendental strictly above it; avoids: none by such a path. A body argument's role whose
  // producer has no stored output is refused (its bound values are not fixed). Addend and
  // Multiplicand keep their independence condition over the argument's leaves.
  void checkRoles(std::size_t m, const v::Value& root, const Pattern& at, std::uint64_t r, std::uint64_t c,
                  const Json& where) {
    const auto& tn = t_.nodes[m];
    if (tn.primitive) return;
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
            if (arg.body && !t_.nodes[arg.ref].out) {
              auto d = where;
              d["argument"] = k;
              d["role"] = name;
              refuse("role_boundary", "a body argument's role needs its producer's output bound to a stored node", d);
            }
            const auto ek = arg.body ? argCut(m, k, at, r, c) : e;
            const auto& rr = arg.body ? bodyReach(root, arg.ref) : inputReach(root);
            if (unite(rr.all, ek) != rr.all) {
              auto d = where;
              d["argument"] = k;
              d["role"] = name;
              d["argument_leaves"] = render(ek);
              d["reached"] = render(rr.all);
              refuse("role", "an element of a role's argument does not reach the atom's output", d);
            }
            // Reaching, and reaching through a transcendental, need a full path; avoiding counts
            // partial paths too.
            const bool through = role == RoleKind::ThroughTranscendental;
            if (unite(rr.full, ek) != rr.full || (through && unite(rr.through, ek) == rr.through &&
                                                  unite(rr.through_full, ek) != rr.through_full)) {
              auto d = where;
              d["argument"] = k;
              d["role"] = name;
              d["argument_leaves"] = render(ek);
              d["reached_in_full"] = render(through ? rr.through_full : rr.full);
              d["partial_uses"] = partial_;
              refuse("partial_operand", "an element of a role's argument reaches the output only through partial uses", d);
            }
            reached = through ? rr.through_full : rr.through;
            ok = through ? unite(reached, ek) == reached : disjoint(reached, ek);
            break;
          }
          case RoleKind::Addend:
            if (arg.body) {  // its leaf-cone walk would cross into the producer (#5030)
              auto d = where;
              d["argument"] = k;
              d["role"] = "addend";
              refuse("role_boundary", "an addend role on a body argument is not supported", d);
            }
            reached = nonAddendLeaves(root, e, memo);
            ok = reached.empty();
            name = "addend";
            break;
          case RoleKind::Multiplicand: {
            if (arg.body) {  // its leaf-cone walk would cross into the producer (#5030)
              auto d = where;
              d["argument"] = k;
              d["role"] = "multiplicand";
              refuse("role_boundary", "a multiplicand role on a body argument is not supported", d);
            }
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
    auto& stored = stored_;
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
    outputs_complete_ = true;
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
      // The cut leaves: each inner node's stored values (by identity), by element.
      std::map<std::string, std::uint64_t> stop;
      if (has_body && !tn.primitive)
        for (const auto& a : tn.args) {
          if (!a.body) continue;
          const auto m = a.ref;
          const auto space = t_.objects.size() + m;
          const auto& values = stored.at(*t_.nodes[m].out);
          for (std::size_t e = 0; e < values.size(); ++e) {
            if (!values[e]) continue;
            const auto k = ident(*values[e]);
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
          requireFull(*slot, actual, fullLeaves(*slot), where);
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
                  x.conversion()->spec->target.element().toString() == tn.store && x.indices().empty())
                x = resolve(x.conversion()->source);
              else if (!x.conversion() && x.indices().empty() &&
                       dynamic_cast<const v::ConvertOp*>(&g_.definingNode(x).operation()) &&
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
            if (!b || !x.indices().empty() || b->opcode() != want->second || tn.args.size() != 2 ||
                g_.type(x).element().kind() != v::ElementKind::Floating ||
                (!tn.type.empty() && op_type != tn.type))
              refuse("primitive", "the stored value is not the term's primitive operation", d);
            // An operand of another element than the op's is converted to the op's.
            const auto convert_to = [&](const std::string& elem, const std::string& named) {
              return !named.empty() ? named : elem == op_type ? std::string() : op_type;
            };
            const auto operands = g_.instanceOperands(x);
            std::vector<std::pair<std::pair<std::size_t, std::uint64_t>, std::pair<std::string, v::Value>>> pending;
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
                  if (!(y.conversion() && y.conversion()->spec->target.element().toString() == cv && y.indices().empty()))
                    return false;
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
                  while (v.conversion() && v.conversion()->spec->kind == v::TypeConversionKind::Bitcast && v.indices().empty()) {
                    const auto e = v.conversion()->spec->target.element().toString();
                    if (e.empty() || (e[0] != 'u' && e[0] != 's' && e[0] != 'b') || e.rfind("bf", 0) == 0) break;
                    v = resolve(v.conversion()->source);
                  }
                  return v;
                };
                // At each conversion level the specs and the indices agree; the base values have one
                // identity (node result, instance, pointer, indices).
                auto z = bits(resolve(*sv));
                y = bits(y);
                while (y.conversion() && z.conversion()) {
                  if (!(*y.conversion()->spec == *z.conversion()->spec) || y.indices().size() != z.indices().size())
                    return false;
                  for (std::size_t i = 0; i < y.indices().size(); ++i)
                    if (ident(y.indices()[i]) != ident(z.indices()[i])) return false;
                  if (static_cast<bool>(y.pointer()) != static_cast<bool>(z.pointer()) ||
                      (y.pointer() && !(*y.pointer() == *z.pointer())))
                    return false;
                  y = bits(resolve(y.conversion()->source));
                  z = bits(resolve(z.conversion()->source));
                }
                return !y.conversion() && !z.conversion() && ident(y) == ident(z);
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
              requireFull(y, leaves(y), fullLeaves(y), d);
              // The element's identity is its coordinate (an unstored node has no layout object).
              const auto coord = coordinate(at, r, c);
              if (!coord || (*coord)[1] >> 32 || (*coord)[0] >> 32)
                unsupported("producer_element", "an unstored inner node read at more than one element or past 2^32", where);
              const std::pair<std::size_t, std::uint64_t> slot_key{a.ref, ((*coord)[0] << 32) | (*coord)[1]};
              // The value's identity (conversion chain and indices included): a and f32(f16(a))
              // are two values.
              const std::pair<std::string, v::Value> value{ident(y), y};
              if (auto it = bound_.find(slot_key); it != bound_.end() && it->second.first != value.first) {
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
              if (auto [it, added] = bound_.emplace(k, v); !added && it->second.first != v.first) {
                auto dc = d;
                dc["inner_node"] = k.first;
                dc["inner_element"] = k.second;
                refuse("producer_binding", "an unstored inner node's element binds two values at two uses", dc);
              }
            }
          } else {
            // 3. Factorization, before the roles: a consumer that does not read its producer's
            // bound values is a factorization failure (a body role's reachability would also
            // refuse it, with a less specific reason).
            if (has_body) {
              const auto expected_cut = materialize(cut, r, c);
              const auto actual_cut = cutter.leaves(*slot);
              if (actual_cut != expected_cut) {
                auto d = where;
                d["expected"] = render(expected_cut);
                d["actual"] = cutter.render(actual_cut);
                refuse("factorization", "with the inner nodes' stored values as leaves, the output reaches other leaves", d);
              }
              cutter.requireFull(*slot, actual_cut, cutter.fullLeaves(*slot), where);
            }
            checkRoles(n, *slot, Pattern{Index{Index::Out, 0}, Index{Index::Out, 1}}, r, c, where);
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
  // Cut values by identity (the cut checker's leaves).
  std::map<std::string, std::uint64_t> stop_;
  std::map<std::string, SetRef> memo_;
  std::map<std::string, Bits> bits_;  // per-bit facts by identity (partial operands)
  std::map<std::string, SetRef> full_memo_;  // fullLeaves by identity
  std::vector<Json> partial_;
  ReachMemo reach_in_;
  std::map<std::size_t, ReachMemo> reach_body_;
  std::map<std::size_t, std::map<std::string, std::uint64_t>> sources_;  // body sources per producer
 public:
  // Stored values per output object element, and per unstored inner node and element the one
  // value its uses bind (`producer_binding`; identity and value). The reference checks bound_
  // against its own reading of the consumers; it keeps its own map of output writes.
  std::map<std::size_t, std::vector<std::optional<v::Value>>> stored_;
  std::map<std::pair<std::size_t, std::uint64_t>, std::pair<std::string, v::Value>> bound_;
  bool outputs_complete_ = false;  // stored_ holds every final output write
 private:
  std::vector<std::string> foreign_;
};

// ---- Reference (tests only, --structural-reference): an independent restatement by plain reachability. ----
// Written apart from the checker: its own successor relation over the value DAG, its own
// structural value equality (resolved at every level), its own term-element expansion (each read
// pattern evaluated at concrete coordinates, recursively through body nodes), its own load mapping
// (memory elimination's read records to term elements by byte arithmetic), its own map from output
// writes to elements, a fresh graph per query and no memo shared across queries. Shared: the
// parsed term, the value graph, memory elimination's records and the scalar specialization
// (SentryWard's infrastructure, not the check's). The checker's producer bindings (bound_) are not
// trusted: each is checked against the consumer's own operands at the coordinate the reference
// computes, and every coordinate a consumer reads must be bound.
// Partial uses: its own rule by concrete evaluation (usesAll, convertsAll; not the checker's
// bits()): an edge into a bit operation, an integer truncation, an integer product, quotient or
// remainder with a fixed operand, float abs, copysign or slct's integer selector is full when
// flipping each bit of the operand that varies between probes of its bit fragment changes the
// result. Leaves, cuts and bindings must be reached along full edges too; a role's element must
// reach the output (through a transcendental, along addend edges, by a guard) along full edges;
// avoids and unguarded paths count every edge. The probes (all free values zero or ones, each
// bit flipped from either) decide a fragment's varying bits exactly when each result bit depends
// on at most two input bits (masks, shifts, resizes, prmt, bfe, bfi); a lop3 chain could hide one.
// Checks: leaves (every output element reaches exactly its term elements), cuts (an opaque node
// with body arguments, with its producers' stored values as stops, reaches exactly its cut
// elements and direct operands), bindings, and the four roles. Verdicts: holds, fails, undecided
// (a step it cannot follow); roles also unsupported (a role the checker refuses by construction:
// addend or multiplicand on a body argument, a role on a body argument of an unstored producer).
class Reference {
 public:
  Reference(const Checker& c, const v::Graph& g, const MemoryEliminationResult& m, const Term& t, bool lifting)
      : c_(c), g_(g), m_(m), t_(t), spec_(g, lifting) {}

  // `matched`: the checker matched, so every binding it needed was made (completeness is checked).
  Json run(bool matched) {
    m_.forEachOutput(g_, [&](const OutputWrite& w) {
      if (const auto e = elementOf(w.region.allocation, w.region.offset)) stored_.emplace(*e, w.value);
    });
    Tally roles, leaves, cuts, bindings;
    std::size_t unsupported_roles = 0;
    const auto attempt = [&](Tally& tally, const Json& where, const auto& f) {
      ++tally.checks;
      try {
        if (f()) ++tally.holds;
        else tally.fail(where);
      } catch (const Undecided& u) {
        auto d = where;
        d["why"] = u.why;
        tally.undecide(d);
      }
    };
    const auto roleChecks = [&](std::size_t m, const v::Value& root, std::uint64_t r, std::uint64_t c) {
      const auto& tn = t_.nodes[m];
      for (std::size_t k = 0; k < tn.args.size(); ++k)
        for (const auto role : tn.roles[k]) {
          const Json where = {{"node", m}, {"element", {r, c}}, {"argument", k}, {"role", int(role)}};
          std::string v;
          attempt(roles, where, [&] {
            v = decide(m, k, role, root, r, c);
            return v != "fails";
          });
          if (v == "unsupported") ++unsupported_roles, --roles.holds;
        }
    };
    for (const auto& [e, value] : stored_) {
      const auto obj = std::get<0>(e), r = std::get<1>(e), c = std::get<2>(e);
      for (std::size_t n = 0; n < t_.nodes.size(); ++n) {
        const auto& tn = t_.nodes[n];
        if (!tn.out || *tn.out != obj) continue;
        const Json where = {{"node", n}, {"element", {r, c}}};
        attempt(leaves, where, [&] { return reachedExactly(value, nullptr, expected(n, r, c, false)); });
        if (tn.primitive) {
          bindingChecks(n, value, r, c, matched, bindings);
          continue;
        }
        bool body = false, stored = true;
        for (const auto& a : tn.args) body |= a.body, stored &= !a.body || t_.nodes[a.ref].out.has_value();
        if (body && stored) attempt(cuts, where, [&] { return reachedExactly(value, &n, expected(n, r, c, true)); });
        roleChecks(n, value, r, c);
      }
    }
    // The checker's bindings: each must be one the reference found (above); their roles.
    for (const auto& [key, value] : c_.bound_) {
      const Json where = {{"node", key.first}, {"coordinate", {key.second >> 32, key.second & 0xffffffffu}}};
      if (!seen_.contains(key)) {
        ++bindings.checks;
        bindings.fail(where);
        continue;
      }
      roleChecks(key.first, value.second, key.second >> 32, key.second & 0xffffffffu);
    }
    auto out = roles.json();
    out["unsupported"] = unsupported_roles;
    out["leaves"] = leaves.json();
    out["cuts"] = cuts.json();
    out["bindings"] = bindings.json();
    return out;
  }

 private:
  struct Undecided {
    std::string why;
  };
  struct Tally {
    std::size_t checks = 0, holds = 0;
    Json fails = Json::array(), undecided = Json::array();
    void fail(const Json& w) { push(fails, w); }
    void undecide(const Json& w) { push(undecided, w); }
    static void push(Json& xs, const Json& w) {
      if (xs.size() < 32) xs.push_back(w);
      else if (xs.size() == 32) xs.push_back("...");
    }
    Json json() const { return {{"checks", checks}, {"holds", holds}, {"fails", fails}, {"undecided", undecided}}; }
  };
  using El = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;  // space, row, column
  using Els = std::set<El>;
  static constexpr std::uint64_t kForeign = ~std::uint64_t{0};
  enum class Kind { Other, Add, Mul, Fma, Bits, Conv, Load, Leaf };
  struct N {
    v::Value x;
    Kind kind = Kind::Other;
    bool trans = false, initial = false;
    std::optional<El> cut;  // a stop: the cut element this value is
    std::size_t indices = 0;  // succ[0, indices): index values
    std::vector<std::size_t> succ;
    std::vector<char> full;   // per succ: the edge reads all of that value's bits (usesAll)
    Els reads;  // an initial load's elements
  };

  // ---- Term elements, by evaluating read patterns at concrete coordinates. ----
  static std::vector<std::array<std::uint64_t, 2>> coords(const Pattern& p, std::uint64_t r, std::uint64_t c) {
    const std::array<std::uint64_t, 2> out{r, c};
    std::array<std::vector<std::uint64_t>, 2> axis;
    for (std::size_t d = 0; d < 2; ++d) {
      const auto& i = p[d];
      switch (i.kind) {
        case Index::Out: axis[d] = {out[i.value]}; break;
        case Index::Const: axis[d] = {i.value}; break;
        case Index::Mod: axis[d] = {out[i.value] % i.size}; break;
        case Index::All:
          for (std::uint64_t u = 0; u < i.value; ++u) axis[d].push_back(u);
          break;
        case Index::Block:
          for (std::uint64_t u = out[i.value] / i.size * i.size, e = u + i.size; u < e; ++u) axis[d].push_back(u);
          break;
      }
    }
    std::vector<std::array<std::uint64_t, 2>> xs;
    for (const auto a : axis[0])
      for (const auto b : axis[1]) xs.push_back({a, b});
    return xs;
  }
  // The elements argument `a` reads at its node's output (r, c): input elements; for a body
  // argument the producer's cut elements (`cut`) or, recursively, the producer's own elements.
  void argElements(const Arg& a, std::uint64_t r, std::uint64_t c, bool cut, Els& out) const {
    for (const auto& p : a.reads)
      for (const auto& q : coords(p, r, c)) {
        if (!a.body) out.insert({t_.inputs[a.ref], q[0], q[1]});
        else if (cut) out.insert({t_.objects.size() + a.ref, q[0], q[1]});
        else for (const auto& e : expected(a.ref, q[0], q[1], false)) out.insert(e);
      }
  }
  Els expected(std::size_t n, std::uint64_t r, std::uint64_t c, bool cut) const {
    Els out;
    for (const auto& a : t_.nodes[n].args) argElements(a, r, c, cut, out);
    return out;
  }

  // ---- Memory: bytes of an allocation to term elements. ----
  std::optional<El> elementOf(const std::string& allocation, std::uint64_t byte) const {
    for (std::size_t i = 0; i < t_.objects.size(); ++i) {
      const auto& o = t_.objects[i];
      if (o.allocation != allocation || byte < o.base) continue;
      const auto rel = byte - o.base, row = rel / o.stride, col = rel % o.stride / o.bytes;
      if (row < o.rows && col < o.cols) return El{i, row, col};
    }
    return std::nullopt;
  }
  Els region(const std::string& allocation, std::uint64_t offset, std::uint64_t bytes) const {
    Els s;
    for (auto b = offset; b < offset + bytes; ++b) {
      const auto e = elementOf(allocation, b);
      s.insert(e ? *e : El{kForeign, 0, b});
    }
    return s;
  }
  // The term elements initial-memory load x reads (none: it reads a stored value).
  std::optional<Els> loadElements(const v::Value& x) {
    const MemoryInstance key{x.nodeId(), x.instanceIndex()};
    const std::uint64_t bytes = g_.type(x.withoutSelection()).element().width() / 8;
    std::optional<std::uint64_t> sel;
    if (!x.indices().empty()) {
      const auto k = x.indices().size() == 1 ? spec_.known(x.indices()[0]) : std::nullopt;
      if (!k || k->literal.empty() || k->literal[0] == '-') throw Undecided{"a load component that is not a constant"};
      sel = std::stoull(k->literal) * bytes;
    }
    const auto narrow = [&](const ConstantRegion& r) {
      return sel ? region(r.allocation, r.offset + *sel, bytes) : region(r.allocation, r.offset, r.bytes);
    };
    if (const auto it = m_.symbolic_initial_reads.find(key); it != m_.symbolic_initial_reads.end()) return narrow(it->second);
    if (m_.symbolic_read_positions.contains(key)) throw Undecided{"symbolic read position"};
    if (const auto* r = m_.readReplacement(key)) {
      if (std::get_if<v::Value>(r)) return std::nullopt;
      if (const auto* i = std::get_if<InitialMemorySymbol>(r)) return narrow(i->region);
      if (sel) throw Undecided{"a component of a reassembled read"};
      Els s;
      const auto piece = [&](const ReinterpretedRead& p) {
        const auto* i = std::get_if<InitialMemorySymbol>(&p.source);
        if (!i) throw Undecided{"read reassembled from stored values"};
        const auto e = region(i->region.allocation, i->region.offset + p.byte_offset, p.type.bitSize() / 8);
        s.insert(e.begin(), e.end());
      };
      if (const auto* p = std::get_if<ReinterpretedRead>(r)) piece(*p);
      else for (const auto& p : std::get<AssembledRead>(*r).pieces) piece(p);
      return s;
    }
    if (const auto i = m_.initialRead(key)) return narrow(i->region);
    throw Undecided{"unresolved load"};
  }

  // ---- Partial uses, by concrete evaluation (the reference's own rule, not bits()). ----
  // A value of the bit fragment (integer bit operations with fixed controls, integer resizes,
  // same-width bitcasts) is evaluated on concrete words for its free values (any other value; a
  // fixed one is its constant). A value's varying bits are those that differ between probes: all
  // free values zero, all ones, and each with one bit flipped from either. An edge is full when
  // flipping each varying bit of the operand (from zero and from ones) changes the operation's
  // result, every control is fixed, and (and, or, lop3) no other operand varies at a bit where it
  // does; partial otherwise.
  __extension__ typedef unsigned __int128 U128;
  static std::uint64_t wmask(unsigned w) { return w >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << w) - 1; }
  unsigned widthOf(const v::Value& x) {
    const auto& t = g_.type(x);
    if (!t.isScalar() || t.element().isParameterized() || t.element().width() > 64) return 0;
    return t.element().width();
  }
  bool bitLike(const v::Value& x) { return g_.type(x).element().isBitLike(); }
  std::optional<std::uint64_t> word(const v::Value& x) {
    const auto w = widthOf(x);
    if (!w || !bitLike(x)) return std::nullopt;
    const auto k = spec_.known(x);
    if (!k) return std::nullopt;
    const auto& s = k->literal;
    if (s == "true") return 1;
    if (s == "false") return 0;
    if (s.empty()) return std::nullopt;
    std::uint64_t u = 0;
    const bool neg = s[0] == '-';
    for (std::size_t i = neg; i < s.size(); ++i) {
      if (s[i] < '0' || s[i] > '9') return std::nullopt;
      u = u * 10 + std::uint64_t(s[i] - '0');
    }
    return (neg ? ~u + 1 : u) & wmask(w);
  }
  bool hasMode(const v::PtxScalarOp& p) {
    for (const auto& m : p.attributes().modifiers)
      if (m != "b32" && m != "u32" && m != "s32" && m != "b64" && m != "u64" && m != "s64" && m != "l" && m != "r" &&
          m != "clamp" && m != "wrap")
        return true;
    return false;
  }
  // The control operand indices of a fragment operation (they must be fixed).
  std::vector<std::size_t> controls(const v::Operation& op) {
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op))
      if (b->opcode() == v::BinaryOpcode::ShiftLeft || b->opcode() == v::BinaryOpcode::ShiftRight) return {1};
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      switch (p->opcode()) {
        case O::BitExtract: return {1, 2};
        case O::BitInsert: return {2, 3};
        case O::BytePermute: case O::FunnelShift: return {2};
        case O::Logic3: return {3};
        default: break;
      }
    }
    return {};
  }
  // Whether resolved x is a fragment value (evaluated, not free).
  bool fragment(const v::Value& x) {
    if (!widthOf(x) || !bitLike(x) || !x.indices().empty()) return false;
    if (const auto& c = x.conversion()) {
      if (!widthOf(c->source) || !bitLike(c->source)) return false;
      if (c->spec->kind == v::TypeConversionKind::Bitcast) return widthOf(c->source) == widthOf(x);
      return !c->spec->attributes.saturate;
    }
    const auto& op = g_.definingNode(x).operation();
    const auto xs = g_.instanceOperands(x).toVector();
    for (const auto i : controls(op))
      if (i >= xs.size() || !word(xs[i])) return false;
    for (const auto& a : xs)
      if (!widthOf(a) || !bitLike(a)) return false;
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op)) {
      using B = v::BinaryOpcode;
      const auto c = b->opcode();
      return c == B::BitAnd || c == B::BitOr || c == B::BitXor || c == B::ShiftLeft || c == B::ShiftRight;
    }
    if (const auto* u = dynamic_cast<const v::UnaryOp*>(&op)) return u->opcode() == v::UnaryOpcode::BitNot;
    if (const auto* cv = dynamic_cast<const v::ConvertOp*>(&op)) return !cv->attributes().saturate;
    if (dynamic_cast<const v::BitcastOp*>(&op)) return widthOf(xs.at(0)) == widthOf(x);
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      const auto c = p->opcode();
      return (c == O::BitExtract || c == O::BitInsert || c == O::BytePermute || c == O::Logic3 || c == O::FunnelShift) &&
             !hasMode(*p);
    }
    return false;
  }
  static std::uint64_t bit(std::uint64_t x, unsigned i) { return i < 64 ? x >> i & 1 : 0; }
  // The result word of operation `op` (result width w, signed result) on operand words xs (widths ws).
  std::optional<std::uint64_t> apply(const v::Operation& op, unsigned w, bool sign, const std::vector<std::uint64_t>& xs,
                                     const std::vector<unsigned>& ws) {
    const auto m = wmask(w);
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op)) {
      using B = v::BinaryOpcode;
      const auto a = xs[0], c = xs[1];
      switch (b->opcode()) {
        case B::BitAnd: return a & c;
        case B::BitOr: return a | c;
        case B::BitXor: return a ^ c;
        case B::ShiftLeft: return c >= w ? 0 : (a << c) & m;
        case B::ShiftRight: {
          if (!sign) return c >= w ? 0 : a >> c;
          const bool neg = bit(a, w - 1);
          if (c >= w) return neg ? m : 0;
          const auto r = a >> c;
          return neg ? (r | (m & ~(m >> c))) & m : r;
        }
        case B::Multiply: {
          if (b->attributes().multiply_mode == v::IntegerMultiplyMode::Low) return (a * c) & m;
          const auto wide = static_cast<U128>(a) * c;
          if (b->attributes().multiply_mode == v::IntegerMultiplyMode::Wide) return std::uint64_t(wide) & m;
          return std::uint64_t(wide >> ws[0]) & m;
        }
        case B::Divide: return c ? a / c : std::optional<std::uint64_t>{};
        case B::Remainder: return c ? a % c : std::optional<std::uint64_t>{};
        default: return std::nullopt;
      }
    }
    if (const auto* u = dynamic_cast<const v::UnaryOp*>(&op)) {
      if (u->opcode() == v::UnaryOpcode::BitNot) return ~xs[0] & m;
      if (u->opcode() == v::UnaryOpcode::Absolute) return xs[0] & (m >> 1);  // float abs: the sign cleared
      return std::nullopt;
    }
    if (dynamic_cast<const v::ConvertOp*>(&op) || dynamic_cast<const v::BitcastOp*>(&op)) return resize(xs[0], ws[0], w, sign);
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      switch (p->opcode()) {
        case O::BitExtract: {
          const auto pos = unsigned(xs[1] & 0xff), len = unsigned(xs[2] & 0xff);
          const bool s = p->sourceType().element().kind() == v::ElementKind::SignedInteger;
          const auto sb = s && len ? bit(xs[0], std::min(pos + len - 1, w - 1)) : 0;
          std::uint64_t r = 0;
          for (unsigned i = 0; i < w; ++i) r |= (i < len && pos + i < w ? bit(xs[0], pos + i) : sb) << i;
          return r;
        }
        case O::BitInsert: {
          const auto pos = unsigned(xs[2] & 0xff), len = unsigned(xs[3] & 0xff);
          auto r = xs[1];
          for (unsigned i = 0; i < len && pos + i < w; ++i) r = (r & ~(std::uint64_t{1} << (pos + i))) | bit(xs[0], i) << (pos + i);
          return r & m;
        }
        case O::BytePermute: {
          const auto both = (xs[1] & 0xffffffffu) << 32 | (xs[0] & 0xffffffffu);
          std::uint64_t r = 0;
          for (unsigned k = 0; k < 4; ++k) {
            const auto s = unsigned(xs[2] >> (4 * k) & 0xf);
            auto byte = both >> (8 * (s & 7)) & 0xff;
            if (s & 8) byte = byte & 0x80 ? 0xff : 0;
            r |= byte << (8 * k);
          }
          return r;
        }
        case O::Logic3: {
          std::uint64_t r = 0;
          for (unsigned i = 0; i < w; ++i)
            r |= bit(xs[3], unsigned(bit(xs[0], i) << 2 | bit(xs[1], i) << 1 | bit(xs[2], i))) << i;
          return r;
        }
        case O::FunnelShift: {
          bool left = false, wrap = false;
          for (const auto& mo : p->attributes().modifiers) left |= mo == "l", wrap |= mo == "wrap";
          const auto n = wrap ? xs[2] & 31 : std::min<std::uint64_t>(xs[2], 32);
          const auto both = static_cast<U128>(xs[1] & 0xffffffffu) << 32 | (xs[0] & 0xffffffffu);
          return std::uint64_t(left ? (both << n) >> 32 : both >> n) & 0xffffffffu;
        }
        case O::CopySign: {  // d = b with a's sign
          const auto s = std::uint64_t{1} << (w - 1);
          return (xs[1] & ~s) | (xs[0] & s);
        }
        case O::SelectSign: return bit(xs[2], ws[2] - 1) ? xs[1] : xs[0];
        case O::MultiplyAdd: case O::MultiplyAddCarry: {
          bool wide = false, high = false;
          for (const auto& mo : p->attributes().modifiers) wide |= mo == "wide", high |= mo == "hi";
          const auto prod = static_cast<U128>(xs[0]) * xs[1];
          const auto pm = high ? std::uint64_t(prod >> ws[0]) : std::uint64_t(prod);
          return (pm + xs[2]) & m;
        }
        default: return std::nullopt;
      }
    }
    return std::nullopt;
  }
  static std::uint64_t resize(std::uint64_t x, unsigned from, unsigned to, bool sign) {
    if (to <= from) return x & wmask(to);
    if (sign && bit(x, from - 1)) return (x | (wmask(to) & ~wmask(from))) & wmask(to);
    return x;
  }
  bool signedSource(const v::Value& x) { return g_.type(x).element().kind() == v::ElementKind::SignedInteger; }
  // The fragment below x: its free values (by structural equality).
  void frees(const v::Value& x0, std::vector<v::Value>& out, unsigned depth = 0) {
    if (depth > 64) throw Undecided{"bit fragment deeper than 64"};
    const auto x = res(x0);
    if (word(x)) return;
    if (!fragment(x)) {
      for (const auto& f : out)
        if (same(f, x)) return;
      if (out.size() > 16) throw Undecided{"bit fragment with more than 16 free values"};
      out.push_back(x);
      return;
    }
    if (x.conversion()) return frees(x.conversion()->source, out, depth + 1);
    for (const auto& a : g_.instanceOperands(x)) frees(a, out, depth + 1);
  }
  std::uint64_t eval(const v::Value& x0, const std::vector<v::Value>& fs, const std::vector<std::uint64_t>& at) {
    const auto x = res(x0);
    if (const auto k = word(x)) return *k;
    if (!fragment(x)) {
      for (std::size_t i = 0; i < fs.size(); ++i)
        if (same(fs[i], x)) return at[i] & wmask(widthOf(x));
      throw Undecided{"a free value outside its fragment"};
    }
    const auto w = widthOf(x);
    if (const auto& c = x.conversion())
      return resize(eval(c->source, fs, at), widthOf(c->source), w, signedSource(c->source));
    const auto& op = g_.definingNode(x).operation();
    std::vector<std::uint64_t> xs;
    std::vector<unsigned> ws;
    const auto operands = g_.instanceOperands(x).toVector();
    for (const auto& a : operands) xs.push_back(eval(a, fs, at)), ws.push_back(widthOf(a));
    const bool sign = dynamic_cast<const v::ConvertOp*>(&op) ? signedSource(operands[0]) : signedSource(x);
    const auto r = apply(op, w, sign, xs, ws);
    if (!r) throw Undecided{"an operation the bit fragment does not evaluate"};
    return *r & wmask(w);
  }
  // The probes of a fragment with free values fs: the base words and each one-bit flip.
  std::vector<std::vector<std::uint64_t>> probes(const std::vector<v::Value>& fs) {
    std::vector<std::vector<std::uint64_t>> out;
    for (const std::uint64_t base : {std::uint64_t{0}, ~std::uint64_t{0}}) {
      const std::vector<std::uint64_t> b(fs.size(), base);
      out.push_back(b);
      for (std::size_t f = 0; f < fs.size(); ++f)
        for (unsigned i = 0; i < std::max(1u, widthOf(fs[f])); ++i) {
          auto p = b;
          p[f] ^= std::uint64_t{1} << i;
          out.push_back(std::move(p));
        }
    }
    return out;
  }
  // The bits of x that differ between probes (all of them for a free value).
  std::uint64_t varying(const v::Value& x0) {
    const auto x = res(x0);
    const auto w = widthOf(x);
    if (word(x)) return 0;
    if (!fragment(x)) return wmask(w);
    std::vector<v::Value> fs;
    frees(x, fs);
    std::uint64_t first = 0, diff = 0;
    bool any = false;
    for (const auto& p : probes(fs)) {
      const auto r = eval(x, fs, p);
      if (any) diff |= r ^ first;
      else first = r, any = true;
    }
    return diff & wmask(w);
  }
  // Whether operand q of x's operation (operands xs) is used in full.
  bool usesAll(const v::Value& x, const v::Operation& op, const std::vector<v::Value>& xs, std::size_t q) {
    if (word(xs[q])) return true;
    if (!x.indices().empty()) return false;  // a component of an operation's matrix result
    const auto ow = widthOf(xs[q]);
    const auto w = widthOf(x);
    // Which operations read part of an operand (else every bit can reach the result).
    bool bits = false, bitwise = false;
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op)) {
      using B = v::BinaryOpcode;
      const auto c = b->opcode();
      bitwise = c == B::BitAnd || c == B::BitOr;
      const bool integer = g_.type(x).element().isInteger();
      if (integer && c == B::Multiply && !word(xs[1 - q])) {
        const auto mode = b->attributes().multiply_mode;
        return factorFull(xs[q], xs[1 - q], mode == v::IntegerMultiplyMode::Wide, mode == v::IntegerMultiplyMode::High);
      }
      if (integer && (c == B::Divide || c == B::Remainder) && !word(xs[1 - q])) return false;  // not decided
      bits = bitwise || c == B::BitXor || c == B::ShiftLeft || c == B::ShiftRight ||
             (integer && (c == B::Multiply || c == B::Divide || c == B::Remainder) && word(xs[1 - q]));
    } else if (const auto* u = dynamic_cast<const v::UnaryOp*>(&op)) {
      bits = u->opcode() == v::UnaryOpcode::Absolute && g_.type(x).element().kind() == v::ElementKind::Floating;
    } else if (dynamic_cast<const v::ConvertOp*>(&op)) {
      bits = bitLike(x) && bitLike(xs[0]) && !dynamic_cast<const v::ConvertOp&>(op).attributes().saturate;
    } else if (dynamic_cast<const v::BitcastOp*>(&op)) {
      if (widthOf(xs[0]) != w || !w) return false;
    } else if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      const auto c = p->opcode();
      bitwise = c == O::Logic3;
      if ((c == O::MultiplyAdd || c == O::MultiplyAddCarry) && q < 2 && g_.type(x).element().isInteger() && !word(xs[1 - q])) {
        bool wide = false, high = false;
        for (const auto& mo : p->attributes().modifiers) wide |= mo == "wide", high |= mo == "hi";
        return factorFull(xs[q], xs[1 - q], wide, high);
      }
      bits = c == O::BitExtract || c == O::BitInsert || c == O::BytePermute || c == O::Logic3 || c == O::FunnelShift ||
             c == O::CopySign || (c == O::SelectSign && q == 2 && g_.type(xs[2]).element().isInteger()) ||
             ((c == O::MultiplyAdd || c == O::MultiplyAddCarry) && q < 2 && g_.type(x).element().isInteger() && word(xs[1 - q]));
      if (bits && c != O::CopySign && c != O::SelectSign && hasMode(*p)) return false;
    }
    if (!bits) return true;
    if (!ow || !w) return false;
    const auto cs = controls(op);
    if (std::find(cs.begin(), cs.end(), q) != cs.end()) return false;  // a control that is data
    for (const auto i : cs)
      if (!word(xs[i])) return false;
    const auto vq = varying(xs[q]);
    std::vector<std::uint64_t> base;
    std::vector<unsigned> ws;
    for (std::size_t i = 0; i < xs.size(); ++i) {
      ws.push_back(widthOf(xs[i]));
      if (i == q) { base.push_back(0); continue; }
      if (const auto k = word(xs[i])) { base.push_back(*k); continue; }
      // Another data operand: for and, or and lop3 it must not vary where q does; its value at
      // the other bits does not matter (per bit). It is taken at its all-zero probe.
      if (bitwise && (varying(xs[i]) & vq)) return false;
      std::vector<v::Value> fs;
      frees(xs[i], fs);
      base.push_back(eval(xs[i], fs, std::vector<std::uint64_t>(fs.size(), 0)));
    }
    if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op); p && p->opcode() == v::PtxScalarOpcode::SelectSign)
      base[0] = 0, base[1] = wmask(w);  // two distinct values, so the selector's effect shows
    const bool sign = dynamic_cast<const v::ConvertOp*>(&op) ? signedSource(xs[0]) : signedSource(x);
    std::uint64_t ignored = 0;  // varying operand bits some flip leaves the result unchanged at
    for (unsigned i = 0; i < ow; ++i) {
      if (!(vq >> i & 1)) continue;
      for (const std::uint64_t from : {std::uint64_t{0}, wmask(ow)}) {
        auto a = base, b = base;
        a[q] = from & ~(std::uint64_t{1} << i);
        b[q] = from | (std::uint64_t{1} << i);
        const auto ra = apply(op, w, sign, a, ws), rb = apply(op, w, sign, b, ws);
        if (!ra || !rb) return false;
        if (*ra == *rb) ignored |= std::uint64_t{1} << i;
      }
    }
    if (!ignored) return true;
    // A float's own bits with only low mantissa bits ignored: a rounding toward zero to a narrower
    // float (sign and exponent kept), numerics inside the atom.
    // Follow same-width bitcasts down to a float.
    std::optional<v::Value> f = res(xs[q]);
    for (unsigned n = 0; f && n < 16 && g_.type(*f).element().kind() != v::ElementKind::Floating; ++n) {
      const auto y = *f;
      if (y.conversion() && y.conversion()->spec->kind == v::TypeConversionKind::Bitcast && y.indices().empty())
        f = res(y.conversion()->source);
      else if (!y.conversion() && y.indices().empty() && dynamic_cast<const v::BitcastOp*>(&g_.definingNode(y).operation()))
        f = res(g_.instanceOperands(y).toVector().at(0));
      else
        f.reset();
      if (f && widthOf(*f) != ow) f.reset();
    }
    if (!f || widthOf(*f) != ow || g_.type(*f).element().kind() != v::ElementKind::Floating) return false;
    const auto t = g_.type(*f).element().toString();
    const unsigned exponent = t == "f16" ? 5 : t == "bf16" || t == "f32" ? 8 : t == "f64" ? 11 : 64;
    unsigned kept = 0;  // the high bits no flip ignores
    while (kept < ow && !(ignored >> (ow - 1 - kept) & 1)) ++kept;
    return ignored == wmask(ow - kept) && kept >= 1 + exponent;
  }
  // Whether an integer product reads every varying bit of factor f, the other factor o not being a
  // word. Low half: f's bit i reaches the product exactly when o has fewer than w - i trailing
  // zeros, for each value o takes; so o's lowest bits up to its first one must not vary (else
  // undecided), and f may vary only below w - t (t: o's trailing zeros at its fixed low bits).
  // Wide: o must have a bit fixed at one (never zero). High half: not decided.
  bool factorFull(const v::Value& f, const v::Value& o, bool wide, bool high) {
    if (high) return false;
    const auto wo = widthOf(o), wf = widthOf(f);
    if (!wo || !wf) return false;
    const auto vo = varying(o);
    std::vector<v::Value> fs;
    frees(o, fs);
    const auto at = eval(o, fs, std::vector<std::uint64_t>(fs.size(), 0));  // o's fixed bits
    if (wide) return (at & ~vo & wmask(wo)) != 0;
    for (unsigned t = 0; t < wo; ++t) {
      if (bit(vo, t)) return false;
      if (bit(at, t)) return t == 0 || (varying(f) & ~wmask(t >= wf ? 0 : wf - t)) == 0;
    }
    return false;
  }
  // Whether the conversion step of x reads all of its source.
  bool convertsAll(const v::Value& x) {
    const auto& c = *x.conversion();
    if (word(c.source)) return true;
    if (!x.indices().empty()) return false;
    const auto w = widthOf(x.withoutSelection()), sw = widthOf(c.source);
    if (c.spec->kind == v::TypeConversionKind::Bitcast) return g_.type(x.withoutSelection()).bitSize() == g_.type(c.source).bitSize();
    if (!bitLike(x) || !bitLike(c.source) || c.spec->attributes.saturate) return true;
    if (!w || !sw) return false;
    return w >= sw || (varying(c.source) & ~wmask(w)) == 0;
  }

  // ---- The value DAG. ----
  v::Value res(v::Value x) { return spec_.selected(g_.canonicalInterpretation(x)); }
  bool same(v::Value a, v::Value b) {
    a = res(a);
    b = res(b);
    if (a.threadSelector() || b.threadSelector()) throw Undecided{"thread selector"};
    if (a.graphIdentity() != b.graphIdentity() || a.nodeId() != b.nodeId() || a.resultIndex() != b.resultIndex() ||
        a.instanceIndex() != b.instanceIndex())
      return false;
    if (bool(a.pointer()) != bool(b.pointer()) || (a.pointer() && !(*a.pointer() == *b.pointer()))) return false;
    if (a.indices().size() != b.indices().size()) return false;
    for (std::size_t i = 0; i < a.indices().size(); ++i)
      if (!same(a.indices()[i], b.indices()[i])) return false;
    if (bool(a.conversion()) != bool(b.conversion())) return false;
    if (!a.conversion()) return true;
    return *a.conversion()->spec == *b.conversion()->spec && same(a.conversion()->source, b.conversion()->source);
  }

  // The successors of resolved x: index values, then the conversion source, the stored value a
  // load reads, or the operation's operands; n.full, whether each edge is a full use.
  std::vector<v::Value> successors(const v::Value& x, N& n) {
    std::vector<v::Value> out(x.indices().begin(), x.indices().end());
    n.indices = out.size();
    n.full.assign(out.size(), 1);
    if (x.conversion()) {
      n.kind = Kind::Conv;
      out.push_back(x.conversion()->source);
      n.full.push_back(convertsAll(x));
      return out;
    }
    const auto& op = g_.definingNode(x).operation();
    if (dynamic_cast<const v::LoadOp*>(&op)) {
      const MemoryInstance key{x.nodeId(), x.instanceIndex()};
      if (const auto* r = m_.readReplacement(key); r && std::get_if<v::Value>(r) && !m_.symbolic_initial_reads.contains(key)) {
        const auto& value = std::get<v::Value>(*r);
        n.kind = Kind::Load;
        out.push_back(x.indices().empty() ? value : g_.component(value, {x.indices().begin(), x.indices().end()}));
        n.full.push_back(1);
        return out;
      }
      // A read of a whole stored value under another type (same bits): that value. A part of one
      // is not followed (its per-byte dependencies are not decided).
      if (const auto* r = m_.readReplacement(key); r && x.indices().empty() && !m_.symbolic_initial_reads.contains(key))
        if (const auto* p = std::get_if<ReinterpretedRead>(r))
          if (const auto* stored = std::get_if<v::Value>(&p->source)) {
            if (p->byte_offset != 0 || p->type.bitSize() != g_.type(*stored).bitSize())
              throw Undecided{"a read of part of a stored value"};
            n.kind = Kind::Load;
            out.push_back(*stored);
            n.full.push_back(1);
            return out;
          }
      n.kind = Kind::Leaf;
      n.initial = true;
      n.reads = *loadElements(x);
      return out;
    }
    if (dynamic_cast<const v::ConstantOp*>(&op) || dynamic_cast<const v::InputParameterOp*>(&op) ||
        dynamic_cast<const v::CoordinateOp*>(&op)) {
      n.kind = Kind::Leaf;
      return out;
    }
    if (dynamic_cast<const v::OpaqueValueOp*>(&op) || dynamic_cast<const v::FailedClcResponseOp*>(&op))
      throw Undecided{"undetermined value"};
    if (const auto* b = dynamic_cast<const v::BinaryOp*>(&op)) {
      using B = v::BinaryOpcode;
      const auto c = b->opcode();
      n.kind = c == B::Add ? Kind::Add : c == B::Multiply ? Kind::Mul
             : (c == B::ShiftLeft || c == B::ShiftRight || c == B::BitAnd || c == B::BitOr) ? Kind::Bits : Kind::Other;
    } else if (dynamic_cast<const v::FmaOp*>(&op)) {
      n.kind = Kind::Fma;
    } else if (dynamic_cast<const v::IdealExp2Op*>(&op)) {
      n.trans = true;
    } else if (const auto* p = dynamic_cast<const v::PtxScalarOp*>(&op)) {
      using O = v::PtxScalarOpcode;
      const auto c = p->opcode();
      n.trans = c == O::Exp2 || c == O::Log2 || c == O::Sin || c == O::Cos || c == O::Tanh;
    } else if (!(dynamic_cast<const v::UnaryOp*>(&op) || dynamic_cast<const v::ConvertOp*>(&op) ||
                 dynamic_cast<const v::BitcastOp*>(&op) || dynamic_cast<const v::CompareOp*>(&op) ||
                 dynamic_cast<const v::SelectOp*>(&op) || dynamic_cast<const v::ThreadForwardOp*>(&op))) {
      throw Undecided{"operation " + std::string(op.name())};
    }
    const auto xs = g_.instanceOperands(x).toVector();
    for (std::size_t q = 0; q < xs.size(); ++q) out.push_back(xs[q]), n.full.push_back(usesAll(x, op, xs, q));
    return out;
  }

  // The DAG below root, values interned by structural equality (root is node 0). With `cuts` (an
  // opaque node), a value equal to a stored value of one of the node's body producers is a stop.
  std::vector<N> graph(const v::Value& root, const std::size_t* cuts = nullptr) {
    std::vector<std::pair<v::Value, El>> stops;
    if (cuts)
      for (const auto& a : t_.nodes[*cuts].args) {
        if (!a.body) continue;
        const auto obj = *t_.nodes[a.ref].out;
        for (const auto& [e, value] : stored_)
          if (std::get<0>(e) == obj) stops.push_back({value, {t_.objects.size() + a.ref, std::get<1>(e), std::get<2>(e)}});
      }
    std::vector<N> ns;
    std::vector<std::size_t> todo;
    const auto intern = [&](const v::Value& x) {
      for (std::size_t i = 0; i < ns.size(); ++i)
        if (same(ns[i].x, x)) return i;
      if (ns.size() > 100000) throw Undecided{"graph budget"};
      N n;
      n.x = res(x);
      ns.push_back(std::move(n));
      todo.push_back(ns.size() - 1);
      return ns.size() - 1;
    };
    intern(root);
    while (!todo.empty()) {
      const auto i = todo.back();
      todo.pop_back();
      N n;
      n.x = ns[i].x;
      for (const auto& [value, e] : stops)
        if (same(n.x, value)) {
          if (n.cut && *n.cut != e) throw Undecided{"one value stored at two cut elements"};
          n.cut = e;
        }
      if (!n.cut)
        for (const auto& y : successors(n.x, n)) n.succ.push_back(intern(y));
      ns[i] = std::move(n);
    }
    return ns;
  }
  // The elements root reaches: initial-load elements, and with `cuts` the cut elements of stops
  // (nothing below a stop). With `full`, along full edges only.
  Els reached(const v::Value& root, const std::size_t* cuts, bool full = false) {
    const auto ns = graph(root, cuts);
    std::vector<char> seen(ns.size(), 0);
    std::vector<std::size_t> todo{0};
    seen[0] = 1;
    Els s;
    while (!todo.empty()) {
      const auto i = todo.back();
      todo.pop_back();
      const auto& n = ns[i];
      if (n.cut) s.insert(*n.cut);
      if (n.initial) s.insert(n.reads.begin(), n.reads.end());
      for (std::size_t q = 0; q < n.succ.size(); ++q)
        if ((!full || n.full[q]) && !seen[n.succ[q]]) seen[n.succ[q]] = 1, todo.push_back(n.succ[q]);
    }
    return s;
  }
  // Every element root reaches, it reaches along full edges, and those are `want`.
  bool reachedExactly(const v::Value& root, const std::size_t* cuts, const Els& want) {
    return reached(root, cuts) == want && reached(root, cuts, true) == want;
  }

  // Primitive consumer n's stored value at (r, c): each body argument of an unstored producer
  // reads one coordinate q of it; (producer, q) must be bound (when the checker matched), to a
  // value that is an operand of n's operation there (below the store conversion), with the
  // producer's elements at q.
  void bindingChecks(std::size_t n, const v::Value& value, std::uint64_t r, std::uint64_t c, bool matched, Tally& t) {
    const auto& tn = t_.nodes[n];
    for (std::size_t k = 0; k < tn.args.size(); ++k) {
      const auto& a = tn.args[k];
      if (!a.body || t_.nodes[a.ref].out || a.reads.size() != 1) continue;
      const auto qs = coords(a.reads[0], r, c);
      if (qs.size() != 1) continue;  // the checker refuses (producer_element)
      const auto q = qs[0];
      const std::pair<std::size_t, std::uint64_t> key{a.ref, (q[0] << 32) | q[1]};
      const Json where = {{"node", n}, {"element", {r, c}}, {"argument", k}, {"producer", a.ref}, {"coordinate", q}};
      const auto it = c_.bound_.find(key);
      if (it == c_.bound_.end()) {
        if (matched) ++t.checks, t.fail(where);
        continue;
      }
      seen_.insert(key);
      ++t.checks;
      try {
        auto x = res(value);
        while (x.conversion() && x.indices().empty() &&
               (x.conversion()->spec->kind == v::TypeConversionKind::Convert ||
                g_.type(x.conversion()->source).element().width() == g_.type(x).element().width()))
          x = res(x.conversion()->source);
        if (!x.conversion() && x.indices().empty() && dynamic_cast<const v::ConvertOp*>(&g_.definingNode(x).operation()))
          x = res(g_.instanceOperands(x).at(0));
        bool operand = false;
        for (const auto& o : g_.instanceOperands(x)) operand |= same(o, it->second.second);
        if (operand && reachedExactly(it->second.second, nullptr, expected(a.ref, q[0], q[1], false))) ++t.holds;
        else t.fail(where);
      } catch (const Undecided& u) {
        auto d = where;
        d["why"] = u.why;
        t.undecide(d);
      }
    }
  }

  static bool contains(const Els& s, const El& e) { return s.contains(e); }

  // Role `role` of argument k of opaque node m, whose value at its output (r, c) is root.
  std::string decide(std::size_t m, std::size_t k, Role role, const v::Value& root, std::uint64_t r, std::uint64_t c) {
    const auto& arg = t_.nodes[m].args[k];
    const bool trans_role = role == RoleKind::ThroughTranscendental || role == RoleKind::AvoidsTranscendental;
    if (arg.body && (!trans_role || !t_.nodes[arg.ref].out)) return "unsupported";
    Els own;
    argElements(arg, r, c, arg.body, own);
    const std::vector<El> elements(own.begin(), own.end());
    const auto ns = graph(root);
    const auto n = ns.size();
    // A source of element e: for an input argument an initial load reading e; for a body argument
    // the producer's stored value at e.
    const auto source = [&](std::size_t i, const El& e) {
      if (!arg.body) return ns[i].initial && contains(ns[i].reads, e);
      const auto it = stored_.find({*t_.nodes[arg.ref].out, std::get<1>(e), std::get<2>(e)});
      return it != stored_.end() && same(ns[i].x, it->second);
    };
    // reach[e][i]: a path from node i reaches source element e (a fresh search per element);
    // reach_full[e][i]: one along full edges. Claims that a path exists use full paths; claims
    // that none does (avoids, the multiplicand's unguarded paths) use every path.
    std::vector<std::vector<char>> reach, reach_full;
    for (const auto& e : elements) {
      std::vector<char> src(n, 0);
      for (std::size_t i = 0; i < n; ++i) src[i] = source(i, e);
      for (const bool full : {false, true}) {
        std::vector<signed char> memo(n, -1);
        std::function<bool(std::size_t)> go = [&](std::size_t i) -> bool {
          if (memo[i] >= 0) return memo[i];
          bool hit = src[i];
          for (std::size_t q = 0; q < ns[i].succ.size(); ++q)
            if (!full || ns[i].full[q]) hit = go(ns[i].succ[q]) || hit;
          memo[i] = hit;
          return hit;
        };
        std::vector<char> row(n);
        for (std::size_t i = 0; i < n; ++i) row[i] = go(i);
        (full ? reach_full : reach).push_back(std::move(row));
      }
      if (!reach_full.back()[0]) return "fails";  // the element does not reach the output in full
    }
    // top[i]: node i is reached from the output along full edges.
    std::vector<char> top(n, 0);
    {
      std::vector<std::size_t> todo{0};
      top[0] = 1;
      while (!todo.empty()) {
        const auto i = todo.back();
        todo.pop_back();
        for (std::size_t q = 0; q < ns[i].succ.size(); ++q)
          if (ns[i].full[q] && !top[ns[i].succ[q]]) top[ns[i].succ[q]] = 1, todo.push_back(ns[i].succ[q]);
      }
    }
    const auto any = [&](std::size_t j) {
      for (const auto& row : reach) if (row[j]) return true;
      return false;
    };
    switch (role) {
      case RoleKind::ThroughTranscendental:
      case RoleKind::AvoidsTranscendental: {
        for (std::size_t e = 0; e < reach.size(); ++e) {
          // through: a full path output -> transcendental -> element; avoids: no path at all.
          bool through = false;
          const bool want = role == RoleKind::ThroughTranscendental;
          for (std::size_t i = 0; i < n && !through; ++i)
            if (ns[i].trans && (!want || top[i]))
              for (std::size_t q = 0; q < ns[i].succ.size(); ++q)
                through |= want ? bool(ns[i].full[q] && reach_full[e][ns[i].succ[q]]) : bool(reach[e][ns[i].succ[q]]);
          if (through != want) return "fails";
        }
        return "holds";
      }
      case RoleKind::Addend: {
        // Every path reaches the element through addend edges only, and no add has the element
        // below both operands.
        std::vector<char> live(n, 0);  // reaches some initial load
        std::vector<signed char> lm(n, -1);
        std::function<bool(std::size_t)> lv = [&](std::size_t i) -> bool {
          if (lm[i] >= 0) return lm[i];
          bool h = ns[i].initial;
          for (const auto j : ns[i].succ) h = lv(j) || h;
          lm[i] = h;
          return h;
        };
        for (std::size_t i = 0; i < n; ++i) live[i] = lv(i);
        for (const auto& row : reach)
          for (std::size_t i = 0; i < n; ++i) {
            const auto& x = ns[i];
            const auto ops = x.succ.size() - x.indices;
            std::size_t live_ops = 0;
            for (std::size_t q = x.indices; q < x.succ.size(); ++q) live_ops += live[x.succ[q]];
            if (x.kind == Kind::Add && ops == 2 && row[x.succ[x.indices]] && row[x.succ[x.indices + 1]]) return "fails";
            for (std::size_t q = 0; q < x.succ.size(); ++q) {
              if (!row[x.succ[q]]) continue;
              const bool idx = q < x.indices;
              bool addend = false;
              if (!idx && x.full[q]) {  // a partial use is never an addend edge
                if (x.kind == Kind::Add && ops == 2) addend = true;
                else if (x.kind == Kind::Fma && ops == 3) addend = (q - x.indices) == 2;
                else if (x.kind == Kind::Conv || x.kind == Kind::Load) addend = true;
                else if (x.kind == Kind::Bits) addend = live_ops == 1;
              }
              if (!addend) return "fails";
            }
          }
        return "holds";
      }
      case RoleKind::Multiplicand: {
        // No path reaches the element without an edge from a product to a factor whose other
        // factor reaches no element of the argument.
        const auto guard = [&](std::size_t i, std::size_t q) {
          const auto& x = ns[i];
          if (!(x.kind == Kind::Mul || x.kind == Kind::Fma) || q < x.indices || q - x.indices > 1 ||
              x.succ.size() - x.indices < 2)
            return false;
          const auto other = x.succ[x.indices + (1 - (q - x.indices))];
          return x.full[q] && any(x.succ[q]) && !any(other);
        };
        for (const auto& e : elements) {
          // unguarded[i]: some path from i reaches the element using no guard edge.
          std::vector<signed char> um(n, -1);
          std::function<bool(std::size_t)> ug = [&](std::size_t i) -> bool {
            if (um[i] >= 0) return um[i];
            bool h = ns[i].initial && contains(ns[i].reads, e);
            for (std::size_t q = 0; q < ns[i].succ.size(); ++q)
              if (!guard(i, q)) h = ug(ns[i].succ[q]) || h;
            um[i] = h;
            return h;
          };
          if (ug(0)) return "fails";
        }
        return "holds";
      }
    }
    return "fails";
  }

  const Checker& c_;
  const v::Graph& g_;
  const MemoryEliminationResult& m_;
  const Term& t_;
  ScalarSpecialization spec_;
  std::map<El, v::Value> stored_;  // output writes by element (the reference's own map)
  std::set<std::pair<std::size_t, std::uint64_t>> seen_;  // bindings a consumer element confirmed
};

}  // namespace

std::string checkStructuralTerm(const value_ir::Graph& graph, const MemoryEliminationResult& memory,
                                const std::string& term_json, const UnitLifting* lifting, bool reference) {
  Json report{{"schema", "sentryward.structural-report/v1"}};
  std::optional<Term> term;
  std::optional<Checker> checker;
  try {
    term = parseTerm(Json::parse(term_json));
    report["row"] = term->row;
    if (!memory.complete) unsupported("memory_elimination", "memory elimination is not complete");
    checker.emplace(graph, memory, *term, lifting);
    report.update(checker->run());
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
  // The reference also runs after a refusal (over the output writes and the bindings committed so
  // far), so a test can compare refusals too; it never changes the result.
  if (reference && checker && checker->outputs_complete_) {
    try {
      report["reference"] = Reference(*checker, graph, memory, *term, lifting != nullptr).run(report["result"] == "matched");
    } catch (const std::exception& e) {
      report["reference"] = {{"error", e.what()}};
    }
  }
  return report.dump(1);
}

}  // namespace sentryward::value_analysis
