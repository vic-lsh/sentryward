#pragma once

#include <optional>
#include <memory>
#include <span>

#include "sentryward/algorithm/memory/access.h"
#include "sentryward/algorithm/memory/family_access.h"
#include "sentryward/algorithm/memory/overlap.h"
#include "sentryward/algorithm/memory/conflict_candidates.h"
#include "sentryward/algorithm/base/dynamic_bitset.h"

namespace sentryward::algorithm {

class MemoryStorageFilter {
 public:
  virtual ~MemoryStorageFilter() = default;
  // Clear only accesses proven storage-disjoint from the selected access.
  virtual void retainCompatible(AccessId access, DynamicBitset& candidates) = 0;
};

class MemoryModel {
 public:
  virtual ~MemoryModel() = default;

  // Null declines family classification; callers retain the exact legacy path.
  [[nodiscard]] virtual std::optional<FamilyFootprint> familyFootprint(
      const MemoryAccessFamily&) { return std::nullopt; }

  // The address space, allocation and alias scope shared by every member of a family, when each
  // member's binding names them (its offset may be symbolic); null when any is unknown. Accesses
  // on storage that no write can reach cannot race (the family checker drops such reads).
  [[nodiscard]] virtual std::optional<FamilyStorage> familyStorage(const MemoryAccessFamily&) {
    return std::nullopt;
  }

  // Class mode's cross-unit pass: familyFootprint, plus strided regions with tensor-property
  // lookups (FamilyFootprint::region_axes), which the family checker never receives.
  [[nodiscard]] virtual std::optional<FamilyFootprint> crossUnitFootprint(const MemoryAccessFamily &family) {
    return familyFootprint(family);
  }

  // Class mode's cross-unit pass: whether some member of `a` placed at `x` and some member of `b`
  // placed at `y` overlap, where the footprints' offsets and lookup arguments are already
  // translated to their units. Disjoint covers every member pair; Unknown declines.
  [[nodiscard]] virtual OverlapResult translatedFamilyOverlap(
      const MemoryAccessFamily&, const FamilyFootprint&, const MemoryAccessFamily&, const FamilyFootprint&) {
    return OverlapResult::Unknown("translated family footprints need a symbolic address model",
                                  "translated_family_unsupported");
  }

  // The window form of a family with footprint `footprint` (WindowForm), or null: then the
  // family is not related by its base and keeps the existing path.
  [[nodiscard]] virtual std::optional<WindowForm> windowForm(const MemoryAccessFamily&, const FamilyFootprint&) {
    return std::nullopt;
  }

  // The window form of one translated region of the cross-unit pass (FamilyFootprint with one lookup
  // term, step 0, `bytes` the bucket's residual hull), from the footprint alone: the caller
  // must have a windowForm for every part the region covers. Null: no window form.
  [[nodiscard]] virtual std::optional<WindowForm> regionWindowForm(const FamilyFootprint&) { return std::nullopt; }

  // True only when a trusted field relation (docs/design/distinct-rows.md) makes the two window
  // forms' atoms unequal at every valuation; false keeps the relation-by-base rule unchanged.
  [[nodiscard]] virtual bool distinctRows(const WindowForm&, const WindowForm&) const { return false; }

  // Whether a lookup of `symbol` on input object `object` can have bounds or premises that differ
  // by element (a field policy, an address invariant naming its elements, or a distinct_rows field
  // on the object): then a decision over a translated lookup depends on the absolute element, not
  // only on relative shifts (the cross-unit memo key, docs/design/gdn-multi-request.md A1). True
  // by default, the conservative answer.
  [[nodiscard]] virtual bool lookupVariesByElement(std::string_view /*symbol*/, std::string_view /*object*/) const {
    return true;
  }

  [[nodiscard]] virtual std::optional<MemoryAnalysisError> validate() const {
    return std::nullopt;
  }

  // Queries are symmetric and use one fixed address/constraint context.
  // Overlap establishes an intersection; Unknown leaves it unresolved.
  [[nodiscard]] virtual OverlapResult mayOverlap(const MemoryAccess& first,
                                                const MemoryAccess& second) = 0;

  // Optional coarse filter; IDs must index the supplied access table.
  // The returned filter owns its index and preserves all possible overlaps.
  [[nodiscard]] virtual std::unique_ptr<MemoryStorageFilter> storageFilter(
      MemoryAccessView) { return nullptr; }

  // Null selects the generic HB-first path. The stream must retain all Unknown pairs.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates> conflictCandidates(
      MemoryAccessView) { return nullptr; }

  // Optional exact existence shortcut. Emits a sufficient subset of conflicting
  // pairs oriented by topological rank: all ordered iff no conflict is unordered.
  // Every emitted pair must have known Overlap; duplicates are permitted. Null
  // declines the shortcut. Not suitable for counting pairs or ignoring failures
  // as warnings. Ranks must outlive the returned stream.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates> conflictOrder(
      MemoryAccessView, std::span<const std::size_t>) { return nullptr; }

  // Existence shortcut with pair exemptions. All potentially exempt writes must
  // be supplied. Preserve every conflicting pair involving their storage; other
  // pairs may use sufficient ordering constraints. No Unknown pair may be lost.
  // Complete scans preserve exemption counts; not for exhaustive race counting.
  // Ranks must outlive the stream. Null declines this shortcut.
  [[nodiscard]] virtual std::unique_ptr<MemoryConflictCandidates>
  conflictCandidatesWithExemptions(MemoryAccessView,
      std::span<const std::size_t>, std::span<const AccessId>) { return nullptr; }
};

}  // namespace sentryward::algorithm
