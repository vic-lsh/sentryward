#include "sentryward/frontend/ptx/choice_facts.h"

#include <algorithm>
#include <functional>
#include <map>
#include <string_view>
#include <vector>

namespace sentryward::frontend::ptx {
namespace {
bool Shared(const Instruction &i) {
  return std::any_of(i.modifiers.begin(), i.modifiers.end(),
                     [](const std::string &m) { return m.rfind("shared", 0) == 0; });
}
// The cross-lane class of an instruction, or empty.
std::string_view CrossLane(const Instruction &i) {
  const auto &op = i.opcode;
  if (op == "atom" || op == "red") return "atomic";
  if (op == "bar" || op == "barrier" || op == "mbarrier" || op == "cp.async.mbarrier" ||
      i.semantics.category == "barrier" || i.semantics.category == "mbarrier")
    return "barrier";
  if (op == "shfl" || op == "vote" || op == "match" || op == "redux" || op == "activemask" ||
      op == "elect")
    return "warp";
  if (Shared(i)) return "shared";
  return {};
}
}  // namespace

nlohmann::json ChoiceFacts(const Function &f, const std::set<std::uint32_t> &lines) {
  const auto size = static_cast<std::uint32_t>(f.instructions.size());
  // Basic blocks: leaders at 0, branch targets and after every control transfer.
  std::set<std::uint32_t> leaders{0, size};
  for (std::uint32_t pc = 0; pc < size; ++pc) {
    const auto &i = f.instructions[pc];
    if (i.branch_target && *i.branch_target <= size) leaders.insert(*i.branch_target);
    for (auto t : i.indirect_targets)
      if (t <= size) leaders.insert(t);
    if (i.control != "none" && i.control != "call") leaders.insert(pc + 1);
  }
  std::vector<std::uint32_t> begin(leaders.begin(), leaders.end());
  const auto blocks = static_cast<std::uint32_t>(begin.size() - 1);  // the last leader is `size`
  std::vector<std::uint32_t> block_of(size);
  for (std::uint32_t b = 0; b < blocks; ++b)
    for (auto pc = begin[b]; pc < begin[b + 1]; ++pc) block_of[pc] = b;
  // Successors; the exit is not a block (no edge).
  std::vector<std::vector<std::uint32_t>> succ(blocks);
  for (std::uint32_t b = 0; b < blocks; ++b) {
    const auto &i = f.instructions[begin[b + 1] - 1];
    auto edge = [&](std::uint32_t pc) {
      if (pc < size) succ[b].push_back(block_of[pc]);
    };
    const bool unconditional = !i.predicate;
    if (i.control == "direct_branch" && i.branch_target) {
      edge(*i.branch_target);
      if (!unconditional) edge(begin[b + 1]);
    } else if (i.control == "indirect_branch") {
      for (auto t : i.indirect_targets) edge(t);
      if (i.indirect_targets.empty())  // unknown targets: any block
        for (std::uint32_t c = 0; c < blocks; ++c) succ[b].push_back(c);
    } else if (i.control == "return" || i.control == "exit" || i.control == "trap") {
      if (!unconditional) edge(begin[b + 1]);
    } else {
      edge(begin[b + 1]);
    }
  }
  // Tarjan: a block is in a cycle when its strongly connected component has two or more blocks or
  // it has a self edge.
  std::vector<int> index(blocks, -1), low(blocks, 0), comp(blocks, -1);
  std::vector<bool> on(blocks, false);
  std::vector<std::uint32_t> stack, comp_size;
  int counter = 0;
  std::function<void(std::uint32_t)> strong = [&](std::uint32_t v) {
    index[v] = low[v] = counter++;
    stack.push_back(v);
    on[v] = true;
    for (auto w : succ[v]) {
      if (index[w] < 0) {
        strong(w);
        low[v] = std::min(low[v], low[w]);
      } else if (on[w]) {
        low[v] = std::min(low[v], index[w]);
      }
    }
    if (low[v] == index[v]) {
      const auto id = static_cast<int>(comp_size.size());
      comp_size.push_back(0);
      std::uint32_t w;
      do {
        w = stack.back();
        stack.pop_back();
        on[w] = false;
        comp[w] = id;
        ++comp_size.back();
      } while (w != v);
    }
  };
  for (std::uint32_t b = 0; b < blocks; ++b)
    if (index[b] < 0) strong(b);
  auto cyclic = [&](std::uint32_t b) {
    return comp_size[comp[b]] > 1 ||
           std::find(succ[b].begin(), succ[b].end(), b) != succ[b].end();
  };
  nlohmann::json out{{"lines", nlohmann::json::array()}, {"cross_lane_ops", nlohmann::json::object()},
                     {"calls", 0}};
  for (auto line : lines) {
    bool found = false, in_loop = false, branch = false;
    for (std::uint32_t pc = 0; pc < size; ++pc) {
      const auto &i = f.instructions[pc];
      if (i.line != line) continue;
      found = true;
      in_loop = in_loop || cyclic(block_of[pc]);
      branch = branch || (i.predicate && (i.control == "direct_branch" || i.control == "return" ||
                                          i.control == "exit" || i.control == "trap"));
    }
    // A line with no instruction is reported as in a loop: nothing is known of it.
    out["lines"].push_back({{"line", line}, {"in_loop", in_loop || !found}, {"branch", branch}});
  }
  std::map<std::string, std::uint64_t> ops;
  std::uint64_t calls = 0;
  for (const auto &i : f.instructions) {
    if (auto c = CrossLane(i); !c.empty()) ++ops[std::string(c)];
    if (i.control == "call") ++calls;
  }
  for (const auto &[k, n] : ops) out["cross_lane_ops"][k] = n;
  out["calls"] = calls;
  return out;
}
}  // namespace sentryward::frontend::ptx
