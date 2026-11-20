// IR-level controls of the structural check's value identity (tests/test_structural_native.py).
//
// No PTX the frontend accepts here yields a value with indices (a component selection of a
// matrix-typed result), so these cases build the value IR directly. M is a two-element matrix
// parameter and m_j = component(M, j) (no leaves: every term operand below is a body argument
// with no input reads, so the leaf checks pass for any j). Each case is a pair: a positive one and
// a negative one that differs ONLY in the index of one value, so only value identity (its indices)
// can tell them apart.
//   factorization: P (no arguments) stores p[j] = m_j; C(P) stores y[j] = -(m_k): k = j matches,
//     k = 1 - j must be refused (factorization: y reads the other cut element).
//   operand: P stores p[j] = m_j; add(P, P) stores y[j] = m_j + m_k: k = j matches, k = 1 - j must
//     be refused (primitive_operands).
//   binding: P unstored; add(P, P) stores y[j] = m_j + m_k: k = j matches, k = 1 - j must be
//     refused (producer_binding: one element of P bound to two values).
// Usage: sentryward-structural-ir-controls <case> <positive|negative>; prints the report.
//
// `random SEED COUNT` (tests/test_structural_differential.py): seeded random IR cases with
// indexed selections, one JSON line each (see randomCases below).
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "sentryward/algorithm/base/analysis_scope.h"
#include "sentryward/value_analysis/memory_actions.h"
#include "sentryward/value_analysis/memory_elimination.h"
#include "sentryward/value_analysis/structural.h"
#include "sentryward/value_ir/graph.h"
#include "sentryward/value_ir/operations.h"

namespace v = sentryward::value_ir;
namespace va = sentryward::value_analysis;

namespace {
std::string object(const std::string& name) {
  return R"({"name": ")" + name +
         R"(", "element": "f32", "element_bytes": 4, "base_bytes": 0, "row_stride_bytes": 8, "rows": 1, "cols": 2, "row_indexed": false})";
}
const std::string kIdent = R"([[{"out": 0}, {"out": 1}]])";
std::string term(const std::string& objects, const std::string& nodes) {
  return R"({"schema": "sentryward.structural-term/v2", "row": "test/index", "template": "test/index", "objects": [)" +
         objects + R"(], "inputs": [], "nodes": [)" + nodes + "]}";
}
std::string producer(bool stored) {
  return R"({"op": "P", "atom": {"kind": "opaque", "roles": []}, "format": {"compute": null, "store": "f32"}, "args": [], "out": )" +
         std::string(stored ? R"("p")" : "null") + "}";
}
std::string add() {
  return R"({"op": "add", "atom": {"kind": "primitive", "operation": "add", "commutative": true}, "format": {"compute": null, "store": "f32"}, "args": [{"body": 0, "reads": )" +
         kIdent + R"(}, {"body": 0, "reads": )" + kIdent + R"(}], "out": "y"})";
}
}  // namespace

int randomCases(std::uint64_t seed, std::size_t count);

int main(int argc, char** argv) {
  if (argc == 4 && std::string(argv[1]) == "random") return randomCases(std::stoull(argv[2], nullptr, 0), std::stoull(argv[3]));
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " factorization|operand|binding positive|negative | random SEED COUNT\n";
    return 2;
  }
  sentryward::algorithm::AnalysisScope scope;
  const std::string kase = argv[1];
  const bool negative = std::string(argv[2]) == "negative";
  v::Graph g;
  const auto f32 = v::ElementType::floating(32);
  const auto m = g.add(std::make_unique<v::InputParameterOp>(v::ValueType::matrix(f32, {2}), "m")).result();
  v::Value index[2] = {
      g.add(std::make_unique<v::ConstantOp>(v::ValueType::scalar(v::ElementType::unsignedInteger(32)), "0")).result(),
      g.add(std::make_unique<v::ConstantOp>(v::ValueType::scalar(v::ElementType::unsignedInteger(32)), "1")).result()};
  const auto component = [&](std::size_t j) { return g.component(m, {index[j]}); };
  va::MemoryEliminationResult mem;
  mem.complete = true;
  const auto write = [&](const std::string& object, std::size_t j, const v::Value& value) {
    mem.outputs.push_back({va::ConstantRegion{va::AllocationIdentity("global", object), 4 * j, 4}, value,
                           va::MemoryInstance{value.nodeId(), value.instanceIndex()}});
  };
  std::string t;
  for (std::size_t j = 0; j < 2; ++j) {
    const auto k = negative ? 1 - j : j;
    if (kase == "factorization") {
      write("p", j, component(j));
      write("y", j, g.add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::Negate, v::ValueType::scalar(f32)), {component(k)}).result());
    } else if (kase == "operand" || kase == "binding") {
      if (kase == "operand") write("p", j, component(j));
      write("y", j, g.add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, v::ValueType::scalar(f32)),
                          {component(j), component(k)}).result());
    } else {
      std::cerr << "unknown case " << kase << "\n";
      return 2;
    }
  }
  if (kase == "factorization")
    t = term(object("p") + ", " + object("y"),
             producer(true) + R"(, {"op": "C", "atom": {"kind": "opaque", "roles": [[]]}, "format": {"compute": null, "store": "f32"}, "args": [{"body": 0, "reads": )" +
                 kIdent + R"(}], "out": "y"})");
  else if (kase == "operand")
    t = term(object("p") + ", " + object("y"), producer(true) + ", " + add());
  else
    t = term(object("y"), producer(false) + ", " + add());
  std::cout << va::checkStructuralTerm(g, mem, t, nullptr) << "\n";
  return 0;
}

// ---- Random IR cases (the differential test's indexed-selection generator). ----
// Each case: four output elements (one row); inputs a and b, each read through two matrix loads of
// two elements (a[j] = component(La[j / 2], j % 2), an indexed selection) or, at random, one scalar
// load per element. A random expression recipe is instantiated per element. Shapes:
//   stored: P(a) stored to p; C over a subset of (P, a, b) stored to y, reading p from its
//     register or back through memory (a load replaced by the stored value);
//   unstored: P(a) unstored under two primitive adds y0 = P + b, y1 = P + c.
// Mutations, with the generator's own truth (the case must not match): `index` (one element reads
// a[j ^ 1]: an index differs), `recompute` (C reads a recomputation of p, not the stored value),
// `rebind` (y1 adds a second build of P's expression), `role` (a transcendental role the
// generator's path model says fails). The model counts paths as the roles do (an operand edge is
// a path; a select's condition too). Output per case: {case, shape, mutation, truth, report}.
namespace {
struct Info {
  std::set<std::string> all, thr;  // source tags reached; reached through ex2 above them
};
struct Ex {
  v::Value value;
  Info info;
};
// An expression recipe over leaf slots: op 0 leaf (slot), 1 add, 2 mul, 3 exp2, 4 neg, 5 select
// (kids: then, else, condition operand).
struct Recipe {
  int op = 0;
  std::size_t slot = 0;
  std::vector<Recipe> kids;
};

class Builder {
 public:
  Builder(v::Graph& g, va::MemoryEliminationResult& m) : g_(g), m_(m) {}
  v::Value constant(const v::ValueType& t, const std::string& literal) {
    return g_.add(std::make_unique<v::ConstantOp>(t, literal)).result();
  }
  // A load of `type` from bytes [offset, +bytes) of `object`'s initial contents.
  v::Value load(const v::ValueType& type, const std::string& object, std::uint64_t offset, std::uint64_t bytes) {
    const auto addr = constant(v::ValueType::scalar(v::ElementType::unsignedInteger(64)), std::to_string(offset));
    const auto x = g_.add(std::make_unique<v::LoadOp>(type), {addr}).result();
    m_.reads.emplace(va::MemoryInstance{x.nodeId(), 0},
                     va::InitialMemorySymbol{next_id_++, va::ConstantRegion{va::AllocationIdentity("global", object), offset, bytes}, type});
    return x;
  }
  // A load the eliminator replaced by stored value `v` (a cut read back through memory).
  v::Value reread(const v::Value& v) {
    const auto addr = constant(v::ValueType::scalar(v::ElementType::unsignedInteger(64)), std::to_string(1000 + next_id_++));
    const auto x = g_.add(std::make_unique<v::LoadOp>(v::ValueType::scalar(v::ElementType::floating(32))), {addr}).result();
    m_.reads.emplace(va::MemoryInstance{x.nodeId(), 0}, v);
    return x;
  }
  void write(const std::string& object, std::size_t j, const v::Value& value) {
    m_.outputs.push_back({va::ConstantRegion{va::AllocationIdentity("global", object), 4 * j, 4}, value,
                          va::MemoryInstance{value.nodeId(), value.instanceIndex()}});
  }
  Ex build(const Recipe& r, const std::vector<Ex>& leaves) {
    const auto f32 = v::ValueType::scalar(v::ElementType::floating(32));
    if (r.op == 0) return leaves.at(r.slot);
    std::vector<Ex> k;
    for (const auto& c : r.kids) k.push_back(build(c, leaves));
    Info info;
    for (const auto& e : k) {
      info.all.insert(e.info.all.begin(), e.info.all.end());
      info.thr.insert(e.info.thr.begin(), e.info.thr.end());
    }
    v::Value x;
    switch (r.op) {
      case 1: x = g_.add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, f32), {k[0].value, k[1].value}).result(); break;
      case 2: x = g_.add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Multiply, f32), {k[0].value, k[1].value}).result(); break;
      case 3:
        x = g_.add(std::make_unique<v::IdealExp2Op>(f32), {k[0].value}).result();
        info.thr = info.all;
        break;
      case 4: x = g_.add(std::make_unique<v::UnaryOp>(v::UnaryOpcode::Negate, f32), {k[0].value}).result(); break;
      default: {
        const auto pred = v::ValueType::scalar(v::ElementType::predicate());
        const auto cond = g_.add(std::make_unique<v::CompareOp>(v::ComparePredicate::Greater, pred),
                                 {k[2].value, constant(f32, "0f00000000")}).result();
        x = g_.add(std::make_unique<v::SelectOp>(f32), {cond, k[0].value, k[1].value}).result();
      }
    }
    return {x, info};
  }

 private:
  v::Graph& g_;
  va::MemoryEliminationResult& m_;
  std::size_t next_id_ = 1;
};

struct Rng {
  std::mt19937_64 g;
  double u() { return std::uniform_real_distribution<double>(0, 1)(g); }
  std::size_t pick(std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n - 1)(g); }
};

Recipe recipe(Rng& rng, std::size_t slots, int depth) {
  Recipe r;
  if (depth == 0 || rng.u() < 0.25) {
    r.slot = rng.pick(slots);
    return r;
  }
  const double c = rng.u();
  r.op = c < 0.3 ? 1 : c < 0.55 ? 2 : c < 0.72 ? 3 : c < 0.85 ? 4 : 5;
  const int arity = r.op == 1 || r.op == 2 ? 2 : r.op == 5 ? 3 : 1;
  for (int i = 0; i < arity; ++i) r.kids.push_back(recipe(rng, slots, depth - 1));
  return r;
}
// Every slot appears: fold the missing ones in with add or mul.
Recipe cover(Rng& rng, Recipe r, std::size_t slots) {
  std::set<std::size_t> used;
  std::function<void(const Recipe&)> walk = [&](const Recipe& x) {
    if (x.op == 0) used.insert(x.slot);
    for (const auto& k : x.kids) walk(k);
  };
  walk(r);
  for (std::size_t s = 0; s < slots; ++s)
    if (!used.contains(s)) {
      Recipe leaf;
      leaf.slot = s;
      Recipe y;
      y.op = rng.u() < 0.5 ? 1 : 2;
      y.kids = {std::move(r), leaf};
      r = std::move(y);
    }
  return r;
}

using Json = nlohmann::json;
Json objectJson(const std::string& name) {
  return {{"name", name}, {"element", "f32"}, {"element_bytes", 4}, {"base_bytes", 0}, {"row_stride_bytes", 16},
          {"rows", 1}, {"cols", 4}, {"row_indexed", false}};
}
const Json kIdentJson = Json::parse(R"([[{"out": 0}, {"out": 1}]])");
Json opaqueJson(const std::string& op, Json args, Json roles, Json out) {
  return {{"op", op}, {"atom", {{"kind", "opaque"}, {"roles", std::move(roles)}}},
          {"format", {{"compute", nullptr}, {"store", "f32"}}}, {"args", std::move(args)}, {"out", std::move(out)}};
}
Json addJson(std::size_t input, const std::string& out) {
  return {{"op", "add"}, {"atom", {{"kind", "primitive"}, {"operation", "add"}, {"commutative", true}}},
          {"format", {{"compute", nullptr}, {"store", "f32"}}},
          {"args", {{{"body", 0}, {"reads", kIdentJson}}, {{"input", input}, {"reads", kIdentJson}}}}, {"out", out}};
}
}  // namespace

int randomCases(std::uint64_t seed, std::size_t count) {
  constexpr std::size_t kN = 4;
  for (std::size_t i = 0; i < count; ++i) {
    Rng rng{std::mt19937_64(seed * 1000003u + i)};
    sentryward::algorithm::AnalysisScope scope;
    v::Graph g;
    va::MemoryEliminationResult mem;
    mem.complete = true;
    Builder b(g, mem);
    const auto f32 = v::ElementType::floating(32);
    const auto u32 = v::ValueType::scalar(v::ElementType::unsignedInteger(32));
    const v::Value index[2] = {b.constant(u32, "0"), b.constant(u32, "1")};
    const bool unstored = rng.u() < 0.35;
    const std::vector<std::string> inputs = unstored ? std::vector<std::string>{"a", "b", "c"} : std::vector<std::string>{"a", "b"};
    // Input elements: an indexed selection of a two-element matrix load, or a scalar load.
    std::map<std::string, std::vector<v::Value>> in;
    for (const auto& o : inputs) {
      if (rng.u() < 0.75) {
        for (std::size_t h = 0; h < kN / 2; ++h) {
          const auto pair = b.load(v::ValueType::matrix(f32, {2}), o, 8 * h, 8);
          for (std::size_t k = 0; k < 2; ++k) in[o].push_back(g.component(pair, {index[k]}));
        }
      } else {
        for (std::size_t j = 0; j < kN; ++j) in[o].push_back(b.load(v::ValueType::scalar(f32), o, 4 * j, 4));
      }
    }
    std::string mutation = "none";
    const double mu = rng.u();
    const std::size_t victim = rng.pick(kN);
    // a's element j as the expression reads it (the index mutation reads a[j ^ 1] at the victim).
    const auto aAt = [&](std::size_t j) {
      const auto k = mutation == "index" && j == victim ? j ^ 1 : j;
      return Ex{in["a"][k], {{"a"}, {}}};
    };
    // P's recipe has an operation: a bare leaf would make p the input value itself (then reading a
    // is reading the cut, and a recomputation is the stored value).
    auto p_recipe = cover(rng, recipe(rng, 1, int(rng.pick(3))), 1);
    if (p_recipe.op == 0) {
      Recipe neg;
      neg.op = 4;
      neg.kids = {p_recipe};
      p_recipe = neg;
    }
    bool truth = true;
    Json term = {{"schema", "sentryward.structural-term/v2"}, {"row", "diff/ir"}, {"template", "diff/ir"}};
    Json nodes = Json::array(), objects = Json::array();
    const auto transRole = [&](const Info& info, const std::string& tag, bool wrong) -> Json {
      // The role the model says holds (or, `wrong`, the opposite: it fails).
      const bool through = info.thr.contains(tag);
      if (!info.all.contains(tag)) return Json::array();
      return Json::array({(through != wrong) ? "through_transcendental" : "avoids_transcendental"});
    };
    if (!unstored) {
      if (mu < 0.15) mutation = "index";
      std::vector<Ex> p(kN);
      for (std::size_t j = 0; j < kN; ++j) {
        p[j] = b.build(p_recipe, {aAt(j)});
        b.write("p", j, p[j].value);
      }
      if (mutation == "index") truth = false;
      // C's arguments: a nonempty subset of (P, a, b), shuffled.
      std::vector<std::string> pool{"p", "a", "b"}, args;
      for (const auto& x : pool)
        if (rng.u() < 0.6) args.push_back(x);
      if (args.empty()) args.push_back(pool[rng.pick(3)]);
      std::shuffle(args.begin(), args.end(), rng.g);
      const bool has_p = std::find(args.begin(), args.end(), "p") != args.end();
      if (has_p && mutation == "none" && mu < 0.25) mutation = "recompute", truth = false;
      const bool reread = rng.u() < 0.5;
      const auto c_recipe = cover(rng, recipe(rng, args.size(), int(rng.pick(4))), args.size());
      const bool wrong = mutation == "none" && rng.u() < 0.25;
      std::vector<Info> cinfo(kN);
      for (std::size_t j = 0; j < kN; ++j) {
        std::vector<Ex> leaves;
        for (const auto& x : args) {
          if (x == "p") {
            Ex e = mutation == "recompute" ? b.build(p_recipe, {aAt(j)}) : p[j];
            if (mutation != "recompute" && reread) e.value = b.reread(e.value);
            // Paths below p count too (no cut ends a path): p's own sources, and p as a source.
            auto info = e.info;
            info.all.insert("p");
            leaves.push_back({e.value, info});
          } else {
            leaves.push_back({in[x][j], {{x}, {}}});
          }
        }
        const auto y = b.build(c_recipe, leaves);
        cinfo[j] = y.info;
        b.write("y", j, y.value);
      }
      Json cargs = Json::array(), croles = Json::array();
      for (const auto& x : args) {
        cargs.push_back(x == "p" ? Json{{"body", 0}, {"reads", kIdentJson}}
                                 : Json{{"input", x == "a" ? 0 : 1}, {"reads", kIdentJson}});
        auto r = rng.u() < 0.7 ? transRole(cinfo[0], x, wrong) : Json::array();
        if (wrong && !r.empty()) truth = false;
        croles.push_back(r);
      }
      for (const auto& o : {"a", "b", "p", "y"}) objects.push_back(objectJson(o));
      nodes.push_back(opaqueJson("P", Json::array({{{"input", 0}, {"reads", kIdentJson}}}),
                                 Json::array({rng.u() < 0.5 ? transRole(p[0].info, "a", false) : Json::array()}), "p"));
      nodes.push_back(opaqueJson("C", cargs, croles, "y"));
      term["inputs"] = {"a", "b"};
    } else {
      if (mu < 0.15) mutation = "index";
      else if (mu < 0.3) mutation = "rebind";
      if (mutation != "none") truth = false;
      const bool wrong = mutation == "none" && rng.u() < 0.25;
      Info pinfo;
      for (std::size_t j = 0; j < kN; ++j) {
        const auto t = b.build(p_recipe, {aAt(j)});
        pinfo = t.info;
        const auto t1 = mutation == "rebind" && j == victim ? b.build(p_recipe, {aAt(j)}) : t;
        const auto add = [&](const v::Value& x, const v::Value& y) {
          return g.add(std::make_unique<v::BinaryOp>(v::BinaryOpcode::Add, v::ValueType::scalar(f32)), {x, y}).result();
        };
        b.write("y0", j, add(t.value, in["b"][j]));
        b.write("y1", j, add(t1.value, in["c"][j]));
      }
      auto proles = rng.u() < 0.7 ? transRole(pinfo, "a", wrong) : Json::array();
      if (wrong && !proles.empty()) truth = false;
      for (const auto& o : {"a", "b", "c", "y0", "y1"}) objects.push_back(objectJson(o));
      nodes.push_back(opaqueJson("P", Json::array({{{"input", 0}, {"reads", kIdentJson}}}), Json::array({proles}), nullptr));
      nodes.push_back(addJson(1, "y0"));
      nodes.push_back(addJson(2, "y1"));
      term["inputs"] = {"a", "b", "c"};
    }
    if (mutation == "none" && !truth) mutation = "role";
    term["objects"] = objects;
    term["nodes"] = nodes;
    const auto report = Json::parse(va::checkStructuralTerm(g, mem, term.dump(), nullptr, true));
    std::cout << Json{{"case", i}, {"shape", unstored ? "unstored" : "stored"}, {"mutation", mutation},
                      {"truth", truth}, {"report", report}}.dump() << "\n";
  }
  return 0;
}
