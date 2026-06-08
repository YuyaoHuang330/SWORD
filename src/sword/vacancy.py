from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
from pymatgen.core import Structure
from scipy.spatial import QhullError, Voronoi, cKDTree

def _site_symbol(site) -> str:
    """Get the element symbol of a site, ignoring oxidation states."""
    return site.specie.symbol


def _environment_score(
    candidate_env: list[tuple[float, str]],
    template_env: list[tuple[float, str]],
) -> float:
    """Scores the similarity between two local environments."""
    # np.fromiter is significantly faster than list comprehensions for array casting
    cand_distances = np.fromiter((dist for dist, _ in candidate_env), dtype=float, count=len(candidate_env))
    templ_distances = np.fromiter((dist for dist, _ in template_env), dtype=float, count=len(template_env))

    rmse = float(np.sqrt(np.mean((cand_distances - templ_distances) ** 2)))

    cand_counts = Counter(symbol for _, symbol in candidate_env)
    templ_counts = Counter(symbol for _, symbol in template_env)
    all_keys = set(cand_counts.keys()) | set(templ_counts.keys())
    
    count_penalty = sum(abs(cand_counts.get(key, 0) - templ_counts.get(key, 0)) for key in all_keys) / len(candidate_env)
    short_contact_penalty = max(0.0, float(templ_distances[0] - cand_distances[0]))

    return rmse + 0.25 * count_penalty + 1.5 * short_contact_penalty


class PeriodicPointIndex:
    """Accelerates nearest-neighbor and void queries under periodic boundary conditions."""
    
    def __init__(
        self,
        *,
        structure: Structure,
        base_cart: np.ndarray,
        base_frac: np.ndarray,
        base_symbols: list[str],
        tiled_cart: np.ndarray,
        tiled_base_indices: np.ndarray,
        tiled_symbols: list[str],
        tree: cKDTree,
    ) -> None:
        self.structure = structure
        self.base_cart = base_cart
        self.base_frac = base_frac
        self.base_symbols = base_symbols
        self.tiled_cart = tiled_cart
        self.tiled_base_indices = tiled_base_indices
        self.tiled_symbols = tiled_symbols
        self.tree = tree

    @classmethod
    def from_sites(cls, structure: Structure, sites: list) -> PeriodicPointIndex:
        base_cart = np.array([site.coords for site in sites], dtype=float)
        base_frac = np.array([site.frac_coords for site in sites], dtype=float)
        base_symbols = [_site_symbol(site) for site in sites]

        # Optimized Cartesian tiling
        shifts = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)
        shift_cart = shifts @ structure.lattice.matrix

        tiled_cart = np.concatenate([base_cart + shift for shift in shift_cart], axis=0)
        tiled_base_indices = np.tile(np.arange(len(sites), dtype=int), len(shifts))
        tiled_symbols = base_symbols * len(shifts)

        return cls(
            structure=structure,
            base_cart=base_cart,
            base_frac=base_frac,
            base_symbols=base_symbols,
            tiled_cart=tiled_cart,
            tiled_base_indices=tiled_base_indices,
            tiled_symbols=tiled_symbols,
            tree=cKDTree(tiled_cart),
        )

    def k_unique_environment(
        self,
        frac_coords: np.ndarray,
        k_neighbors: int,
        *,
        skip_base_index: int | None = None,
    ) -> list[tuple[float, str]]:
        cart_coords = self.structure.lattice.get_cartesian_coords(frac_coords)
        query_k = min(len(self.tiled_cart), max(k_neighbors * 4, k_neighbors))

        while True:
            distances, indices = self.tree.query(cart_coords, k=query_k)
            distances = np.atleast_1d(distances)
            indices = np.atleast_1d(indices)

            environment: list[tuple[float, str]] = []
            seen: set[int] = set()
            for distance, tiled_idx in zip(distances.tolist(), indices.tolist()):
                base_idx = int(self.tiled_base_indices[tiled_idx])
                if base_idx == skip_base_index or base_idx in seen:
                    continue
                seen.add(base_idx)
                environment.append((float(distance), self.tiled_symbols[tiled_idx]))
                if len(environment) == k_neighbors:
                    return environment

            if query_k == len(self.tiled_cart):
                return environment
            query_k = min(len(self.tiled_cart), query_k * 2)

    def nearest_unique_distance(self, frac_coords: np.ndarray) -> float:
        return self.k_unique_environment(frac_coords, 1)[0][0]

    def nearest_other_distance(self, base_index: int) -> float:
        return self.k_unique_environment(self.base_frac[base_index], 1, skip_base_index=base_index)[0][0]


def _same_species_nn_stats(species_index: PeriodicPointIndex) -> dict[str, float] | None:
    if len(species_index.base_cart) < 2:
        return None
    nn_distances = [species_index.nearest_other_distance(i) for i in range(len(species_index.base_cart))]
    arr = np.array(nn_distances, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
    }


def _periodic_voronoi_candidates(
    structure: Structure,
    point_index: PeriodicPointIndex,
    *,
    min_void_radius: float,
    merge_tolerance: float,
    k_neighbors: int,
) -> list[dict[str, Any]]:
    if len(point_index.base_cart) < 4:
        return []

    try:
        vor = Voronoi(point_index.tiled_cart)
    except QhullError:
        return []

    raw_candidates: list[dict[str, Any]] = []
    lattice = structure.lattice.matrix
    
    for vertex in vor.vertices:
        frac_coords = np.mod(structure.lattice.get_fractional_coords(vertex), 1.0)
        environment = point_index.k_unique_environment(frac_coords, k_neighbors)
        void_radius = environment[0][0]
        if void_radius < min_void_radius:
            continue
        raw_candidates.append(
            {
                "frac_coords": frac_coords,
                "cart_coords": structure.lattice.get_cartesian_coords(frac_coords),
                "void_radius": float(void_radius),
            }
        )

    deduped: list[dict[str, Any]] = []
    # Sort and filter in one pass
    for candidate in sorted(raw_candidates, key=lambda item: item["void_radius"], reverse=True):
        too_close = False
        for existing in deduped:
            delta = candidate["frac_coords"] - existing["frac_coords"]
            delta -= np.round(delta)
            if np.linalg.norm(delta @ lattice) < merge_tolerance:
                too_close = True
                break
        if not too_close:
            deduped.append(candidate)

    return deduped


def _score_shared_void_pool(
    shared_voids: list[dict[str, Any]],
    *,
    available_species: list[str],
    host_indices: dict[str, PeriodicPointIndex],
    templates: dict[str, list[list[tuple[float, str]]]],
    k_neighbors: int,
) -> list[dict[str, Any]]:
    scored_voids: list[dict[str, Any]] = []
    for void in shared_voids:
        frac_coords = void["frac_coords"]
        species_scores = {}
        for species in available_species:
            candidate_env = host_indices[species].k_unique_environment(frac_coords, k_neighbors)
            species_scores[species] = float(min(_environment_score(candidate_env, template) for template in templates[species]))

        ordered_scores = sorted(species_scores.items(), key=lambda item: item[1])
        best_species, best_score = ordered_scores[0]
        second_best_species, second_best_score = ordered_scores[1]

        scored_voids.append(
            {
                "frac_coords": frac_coords,
                "cart_coords": void["cart_coords"],
                "void_radius": float(void["void_radius"]),
                "all_species_scores": {key: float(val) for key, val in species_scores.items()},
                "best_species": best_species,
                "best_species_score": float(best_score),
                "best_competing_species": second_best_species,
                "best_competing_score": float(second_best_score),
                "species_margin": float(second_best_score - best_score),
            }
        )
    return scored_voids


def _make_filled_structure(structure: Structure, target_symbol: str, frac_coords: np.ndarray) -> Structure:
    filled = structure.copy()
    representative = next(site.specie for site in structure if _site_symbol(site) == target_symbol)
    filled.append(representative, frac_coords, coords_are_cartesian=False, validate_proximity=False)
    return filled


def _make_multi_filled_structure(
    structure: Structure,
    target_symbol: str,
    frac_coords_list: list[np.ndarray],
) -> Structure:
    filled = structure.copy()
    representative = next(site.specie for site in structure if _site_symbol(site) == target_symbol)
    for frac_coords in frac_coords_list:
        filled.append(representative, frac_coords, coords_are_cartesian=False, validate_proximity=False)
    return filled


def _make_global_filled_structure(structure: Structure, filled_sites: list[dict[str, Any]]) -> Structure:
    filled = structure.copy()
    representatives = {
        symbol: next(site.specie for site in structure if _site_symbol(site) == symbol)
        for symbol in {_site_symbol(site) for site in structure}
    }
    for site in filled_sites:
        filled.append(representatives[site["species"]], site["frac_coords"], coords_are_cartesian=False, validate_proximity=False)
    return filled


def _sort_candidates(candidates: list[dict[str, Any]]) -> None:
    candidates.sort(
        key=lambda item: (
            -round(item["margin"], 12),
            round(item["score"], 12),
            tuple(round(x, 12) for x in item["frac_coords"]),
        )
    )


def _build_species_result(
    structure: Structure,
    species: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    _sort_candidates(candidates)
    kept_candidates = candidates[:top_k]
    all_frac_coords = [candidate["frac_coords"] for candidate in candidates]
    return {
        "num_candidates": len(candidates),
        "candidates": kept_candidates,
        "all_filled_frac_coords": all_frac_coords,
        "all_filled_structure": _make_multi_filled_structure(
            structure,
            species,
            [np.array(frac_coords, dtype=float) for frac_coords in all_frac_coords],
        ),
    }


def _finalize_results(structure: Structure, results: dict[str, Any]) -> dict[str, Any]:
    all_filled_sites = [
        {"species": species, "frac_coords": frac_coords}
        for species, payload in sorted(results["species_results"].items())
        for frac_coords in payload.get("all_filled_frac_coords", [])
    ]
    has_fill = bool(all_filled_sites)
    results["all_filled_sites"] = all_filled_sites
    results["has_fill"] = has_fill
    results["all_filled_structure"] = _make_global_filled_structure(structure, all_filled_sites) if has_fill else None
    return results


def find_vacancy_ordered(
    structure: Structure,
    *,
    species_subset: list[str] | None = None,
    k_neighbors: int = 6,
    top_k: int = 4,
    min_void_radius: float = 1.0,
    shared_pool_min_void_radius: float | None = None,
    min_distance_to_existing_same: float = 0.80,
    merge_tolerance: float = 0.40,
    max_geometry_score: float = 0.25,
    min_species_margin: float = 0.20,
    same_species_lower_factor: float = 0.90,
    same_species_upper_factor: float = 1.20,
    only_possible: bool = True,
) -> dict[str, Any]:
    
    available_species = sorted({_site_symbol(site) for site in structure})
    target_species = available_species if species_subset is None else [sp for sp in species_subset if sp in available_species]

    species_sites = {species: [site for site in structure if _site_symbol(site) == species] for species in available_species}
    host_sites = {species: [site for site in structure if _site_symbol(site) != species] for species in available_species}

    shared_index = PeriodicPointIndex.from_sites(structure, list(structure))
    host_indices = {species: PeriodicPointIndex.from_sites(structure, host_sites[species]) for species in available_species}
    same_species_indices = {species: PeriodicPointIndex.from_sites(structure, species_sites[species]) for species in available_species}

    templates = {
        species: [host_indices[species].k_unique_environment(site.frac_coords, k_neighbors) for site in species_sites[species]]
        for species in available_species
    }
    same_species_stats = {species: _same_species_nn_stats(same_species_indices[species]) for species in available_species}

    shared_voids = _periodic_voronoi_candidates(
        structure,
        shared_index,
        min_void_radius=min_void_radius if shared_pool_min_void_radius is None else shared_pool_min_void_radius,
        merge_tolerance=merge_tolerance,
        k_neighbors=k_neighbors,
    )
    scored_voids = _score_shared_void_pool(
        shared_voids,
        available_species=available_species,
        host_indices=host_indices,
        templates=templates,
        k_neighbors=k_neighbors,
    )

    results: dict[str, Any] = {"formula": structure.formula, "num_sites": len(structure), "species_results": {}}

    for target in target_species:
        stats = same_species_stats[target]
        lower_same = None if stats is None else same_species_lower_factor * stats["min"]
        upper_same = None if stats is None else same_species_upper_factor * stats["max"]

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        
        for void in scored_voids:
            frac_coords = void["frac_coords"]
            nearest_same = same_species_indices[target].nearest_unique_distance(frac_coords)
            if nearest_same < min_distance_to_existing_same:
                continue

            same_spacing_ok = True if lower_same is None else lower_same <= nearest_same <= upper_same
            passed = (
                void["best_species"] == target
                and void["best_species_score"] <= max_geometry_score
                and void["species_margin"] >= min_species_margin
                and same_spacing_ok
            )

            record = {
                "frac_coords": [float(x) for x in frac_coords],
                "score": float(void["all_species_scores"][target]),
                "margin": float(void["species_margin"]),
                "nearest_same_distance": float(nearest_same),
            }

            if passed:
                record["filled_structure"] = _make_filled_structure(structure, target, frac_coords)
                candidates.append(record)
            else:
                rejected.append({
                    "best_species": void["best_species"],
                    "score": float(void["all_species_scores"][target]),
                    "margin": float(void["species_margin"]),
                    "same_spacing_ok": same_spacing_ok,
                })

        if candidates:
            results["species_results"][target] = _build_species_result(structure, target, candidates, top_k=top_k)
        elif not only_possible:
            best_rejected = rejected[0] if rejected else None
            reason = (
                "No usable shared-void candidates were generated."
                if best_rejected is None
                else (
                    f"No candidate passed. Best raw void was classified as {best_rejected['best_species']} "
                    f"(score={best_rejected['score']:.3f}, margin={best_rejected['margin']:.3f}, "
                    f"same_spacing_ok={best_rejected['same_spacing_ok']})."
                )
            )
            results["species_results"][target] = {
                "num_candidates": 0,
                "candidates": [],
                "reason": reason,
            }

    return _finalize_results(structure, results)
