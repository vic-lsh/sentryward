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
#include <iostream>
#include <memory>
#include <string>

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

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " factorization|operand|binding positive|negative\n";
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
