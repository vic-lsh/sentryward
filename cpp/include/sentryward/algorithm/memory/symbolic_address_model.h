#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sentryward/algorithm/memory/address_solver.h"
#include "sentryward/algorithm/memory/constant_address_model.h"

namespace sentryward::algorithm {

// The prefix of a TensorPropertySymbol.
inline constexpr std::string_view kTensorPropertySymbolPrefix = "tensor_property:";
// The internal table-lookup symbol of an input object's elements under tensor_properties:
// an address offset term {"operator":"gather","symbol":S,"args":[element byte offset]}.
inline std::string TensorPropertySymbol(std::string_view object, std::string_view element_type) {
  return std::string(kTensorPropertySymbolPrefix) + std::string(object) + ":" + std::string(element_type);
}
// The object and element type of a TensorPropertySymbol (the element type has no ':'), else nullopt.
inline std::optional<std::pair<std::string, std::string>> ParseTensorPropertySymbol(std::string_view symbol) {
  constexpr auto prefix = kTensorPropertySymbolPrefix;
  if (!symbol.starts_with(prefix)) return std::nullopt;
  const auto split = symbol.rfind(':');
  if (split < prefix.size()) return std::nullopt;
  return std::pair{std::string(symbol.substr(prefix.size(), split - prefix.size())),
                   std::string(symbol.substr(split + 1))};
}
// The symbol of an element function of those elements (a canonical operator tree `function`
// over the element value): {"operator":"gather","symbol":S,"args":[element byte offset],
// "object":O,"lower":L,"upper":U}, an exact deterministic value within [L, U].
inline std::string TensorPropertyFunctionSymbol(std::string_view object, std::string_view element_type,
                                                std::string_view function) {
  return "tensor_property_function:" + std::to_string(object.size()) + ":" + std::string(object) + ":" +
         std::string(element_type) + ":" + std::string(function);
}

struct SymbolicSolverOptions final {
  // Budget per pair across its environment and overlap checks; must be positive.
  unsigned timeout_ms = 100;

  friend bool operator==(const SymbolicSolverOptions&, const SymbolicSolverOptions&) = default;
};

struct SymbolicAddressStats final {
  // Pairs the constant model left Unknown and this model refined.
  std::size_t refined = 0;
  std::size_t cache_hits = 0;
  std::size_t layer0_disjoint = 0;
  // Decided from the address-invariant environment or a constant offset
  // difference, without calling the solver.
  std::size_t ground_disjoint = 0;
  std::size_t ground_overlap = 0;
  // The offset difference is a constant plus a multiple of its coefficients'
  // gcd, and no such value lies in the overlap interval.
  std::size_t congruence_disjoint = 0;
  // Two different elements of one injective input object, with opposite
  // coefficients, differ by at least the coefficient.
  std::size_t injective_disjoint = 0;
  std::size_t solver_queries = 0;
  std::size_t solver_unsat = 0;
  std::size_t solver_sat = 0;
  std::size_t solver_unknown = 0;

  friend bool operator==(const SymbolicAddressStats&, const SymbolicAddressStats&) = default;
};

// Containment of one access's byte range in [0, extent) of its allocation (footprint check).
struct ContainmentVerdict final {
  enum class Status { kContained, kViolated, kUnknown };
  Status status = Status::kUnknown;
  std::string address_space;
  std::string allocation;
  // Touched byte interval [min_byte, max_byte_exclusive) of the access when it is bounded:
  // exact for a constant offset, the interval hull over the atom bounds for a symbolic one.
  std::optional<std::int64_t> min_byte, max_byte_exclusive;
  bool constant_offset = false;
  // The offset's constant part and the gcd of its atom coefficients (0 for a constant offset):
  // with the gcd a multiple of a row stride, the offset's column within a row is the constant's.
  std::optional<std::int64_t> constant;
  std::uint64_t coefficient_gcd = 0;
  // Some atom's bounds come from a field property: they hold for this element, not necessarily
  // for the element a CTA-class member reads in its place.
  bool field_bounded = false;
  // What decided the status (docs/design/gdn-multi-request.md, A2): kConstant, the exact constant
  // offset; kPolicyHull, the interval hull of the atoms over their tensor-property policies alone
  // (BoundsOf: object policy tightened by the field policies containing the element); kSolver, the
  // solver over the address-invariant environment (term bounds, joint injectivity, injectivity).
  // kNone: no status was decided (no extent, or kUnknown before either step).
  enum class Basis { kNone, kConstant, kPolicyHull, kSolver };
  Basis basis = Basis::kNone;
  // The offset's atoms as tensor-property lookups (symbol, input object, element byte offset), in
  // atom order; `lookups_complete` false when some atom is not a one-argument lookup of a known
  // input object (then a class member's translated element is undefined).
  struct Lookup {
    std::string symbol, object;
    address_solver::Int element = 0;
  };
  std::vector<Lookup> lookups;
  bool lookups_complete = true;
  // kViolated: which side ("below_zero" or "past_extent") and the offset (constant) or the
  // solver's candidate valuation (symbolic); kUnknown: why.
  std::string side, witness, reason;
};

// Symbolic dynamic-address overlap (dynamic.pdf, layers 0 and 2).
//
// Addresses are allocation-tagged affine forms over interned atoms. Pairs the
// constant model decides (resolved constant footprints, registers, constant
// allocation separation) keep its exact answers and candidate streams. Pairs
// it leaves Unknown are refined by allocation separation (layer 0) and by a
// ground SMT query over the address-invariant environment Gamma (layer 2).
// Disjoint holds for every valuation satisfying Gamma; inconclusive queries
// stay Unknown. See docs/assumptions.md for the binding and invariant formats.
class SymbolicAddressModel final : public MemoryModel {
 public:
  // `address_invariants` is a JSON array of Gamma constraints (default empty).
  // `tensor_properties` are normalized input records (object, element_type, lower_bound,
  // upper_bound, injective); they bound the TensorPropertySymbol lookups (default none).
  // `field_relations` are the input's distinct_rows relations (docs/design/distinct-rows.md).
  explicit SymbolicAddressModel(const std::vector<ConstantAddressBinding>& bindings,
                                std::string_view address_invariants = "[]",
                                SymbolicSolverOptions solver = {},
                                std::size_t cache_capacity = 4096,
                                std::string_view tensor_properties = "[]",
                                std::string_view field_relations = "[]");
  ~SymbolicAddressModel() override;
  SymbolicAddressModel(SymbolicAddressModel&&) noexcept;
  SymbolicAddressModel& operator=(SymbolicAddressModel&&) noexcept;
  SymbolicAddressModel(const SymbolicAddressModel&) = delete;
  SymbolicAddressModel& operator=(const SymbolicAddressModel&) = delete;

  [[nodiscard]] std::optional<MemoryAnalysisError> validate() const override;
  [[nodiscard]] OverlapResult mayOverlap(const MemoryAccess& first,
                                         const MemoryAccess& second) override;
  // Streams come from the constant model: they retain every pair it leaves
  // Unknown, a superset of the pairs this model does not prove disjoint.
  [[nodiscard]] std::unique_ptr<MemoryStorageFilter> storageFilter(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidates(
      MemoryAccessView accesses) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictOrder(
      MemoryAccessView accesses, std::span<const std::size_t> ranks) override;
  [[nodiscard]] std::unique_ptr<MemoryConflictCandidates> conflictCandidatesWithExemptions(
      MemoryAccessView accesses, std::span<const std::size_t> ranks,
      std::span<const AccessId> exempt_writes) override;

  [[nodiscard]] const ConstantAddressModel& constantModel() const noexcept;
  [[nodiscard]] std::optional<FamilyFootprint> familyFootprint(const MemoryAccessFamily&) override;
  [[nodiscard]] std::optional<FamilyStorage> familyStorage(const MemoryAccessFamily&) override;
  // Also a TMA region whose offset's symbolic part is tensor-property lookups (region_axes set).
  [[nodiscard]] std::optional<FamilyFootprint> crossUnitFootprint(const MemoryAccessFamily&) override;
  // Footprints whose symbolic suffix is tensor-property lookups (FamilyFootprint::gathers): each
  // family is its lowest member's address plus one progression axis, decided like a pair of
  // bindings. Unknown when either footprint has another suffix or is not affine.
  [[nodiscard]] OverlapResult translatedFamilyOverlap(const MemoryAccessFamily& a, const FamilyFootprint& x,
                                                      const MemoryAccessFamily& b, const FamilyFootprint& y) override;
  // A family on one resolved allocation without alias scope whose members' offsets have one
  // tensor-property lookup term S x t (the same atom and S > 0 for every member, all of one width),
  // the atom with both premise bounds and 0 <= lower; the footprint's span within [0, S) and
  // S x upper + span < 2^63, in checked 128-bit arithmetic. Otherwise null.
  [[nodiscard]] std::optional<WindowForm> windowForm(const MemoryAccessFamily&, const FamilyFootprint&) override;
  // True when a distinct_rows relation makes the two forms' atoms rows i != j of its write field,
  // or row i of its write field and row j of its read field (either order): then t1 != t2.
  [[nodiscard]] bool distinctRows(const WindowForm&, const WindowForm&) const override;
  // Footprint containment. `extent_bytes` null: only the touched interval is computed. Shapes the
  // address model cannot express (unresolved allocation, TMA regions, non-affine offsets) are
  // kUnknown, never kContained. Symbolic offsets are decided by the address solver over the
  // tensor-property and address-invariant environment, with `timeout_ms` per query. `lower`: the
  // range is [lower, extent_bytes) (a CTA-class representative's range narrowed by its members' δ).
  [[nodiscard]] ContainmentVerdict containment(const MemoryAccess& access,
                                                std::optional<std::uint64_t> extent_bytes,
                                                unsigned timeout_ms = 2000, std::int64_t lower = 0);
  // The indexes of the field policies (tensor_properties with a `field`) of lookup symbol `symbol`
  // whose elements include byte offset `element`, ascending. Equal sets give equal BoundsOf.
  [[nodiscard]] std::vector<std::size_t> fieldPolicySet(std::string_view symbol, address_solver::Int element) const;
  [[nodiscard]] SymbolicAddressStats stats() const noexcept;
  // Whether this build links an SMT backend, and its name ("z3" or "unavailable").
  [[nodiscard]] static bool SolverAvailable() noexcept;
  [[nodiscard]] static const char* SolverName() noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sentryward::algorithm
