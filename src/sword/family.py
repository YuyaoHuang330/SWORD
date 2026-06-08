from __future__ import annotations

import logging
import re
import traceback
from itertools import combinations
from io import StringIO
from typing import TYPE_CHECKING, Any, Union

import pandas as pd
from monty.json import MSONable
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecie
from pymatgen.io.cif import CifParser

from .label import get_sword_label
from .vacancy import find_vacancy_ordered

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

logger = logging.getLogger(__name__)

# Pre-compile regex patterns for significant loop speedups
_ORBIT_PATTERN = re.compile(r"(\d*)(?:\((.*)\)|(.*))")
_MIX_PATTERN = re.compile(r'(?:VAC|[A-Z][a-z]?)(?:\+(?:VAC|[A-Z][a-z]?))+')

class SWORDFamilyMatcher(MSONable):
    """
    Match and classify structures into families based on SWORD dictionaries.
    """

    def __init__(
        self,
        symprec_child: float = 1e-2,
        symprec_search: float = 1.0,
        parser_occ_tolerance: float = 10.0,
        fill_vacancy: bool = True,
    ) -> None:
        self.symprec_child = symprec_child
        self.symprec_search = symprec_search
        self.parser_occ_tolerance = parser_occ_tolerance
        self.fill_vacancy = fill_vacancy

    @staticmethod
    def _fix_parent_label(label: str, mix_str: str) -> str:
        parts = label.split("_")
        new_parts = []
        for p in parts:
            if 'X' in p:
                inner = p.strip("{}")
                items = inner.split(",")
                modified_any = False
                new_items = []
                
                for item in items:
                    m = _ORBIT_PATTERN.fullmatch(item)
                    if not m:
                        new_items.append(item)
                        continue
                        
                    cnt = m.group(1)
                    orbit_str = m.group(2) if m.group(2) is not None else m.group(3)
                    
                    elements_in_item = orbit_str.split("+")
                    if "X" not in elements_in_item:
                        new_items.append(item)
                        continue
                        
                    modified_any = True
                    new_elements = []
                    for el in elements_in_item:
                        if el == "X":
                            new_elements.extend(mix_str.split("+"))
                        else:
                            new_elements.append(el)
                            
                    new_elements.sort()
                    new_orbit_str = "+".join(new_elements)
                    
                    if cnt:
                        if "+" in new_orbit_str:
                            new_items.append(f"{cnt}({new_orbit_str})")
                        else:
                            new_items.append(f"{cnt}{new_orbit_str}")
                    else:
                        new_items.append(new_orbit_str)
            
                if modified_any:
                    new_items.sort()
                    if len(new_items) == 1 and "+" not in new_items[0] and not new_items[0][0].isdigit():
                        new_parts.append(new_items[0])
                    else:
                        new_parts.append("{" + ",".join(new_items) + "}")
                else:
                    new_parts.append(p)
            else:
                new_parts.append(p)
        return "_".join(new_parts)

    def _parse_input(self, data: Union[str, Structure]) -> Structure:
        if isinstance(data, Structure):
            return data.copy()
            
        if "\n" in str(data):
            parser = CifParser(StringIO(data), occupancy_tolerance=self.parser_occ_tolerance)
        else:
            parser = CifParser(data, occupancy_tolerance=self.parser_occ_tolerance)
            
        return parser.parse_structures(primitive=False)[0]

    def _heal_structure(self, structure: Structure) -> None:
        for i, site in enumerate(structure):
            total_occ = sum(site.species.values())
            if 0 < total_occ < 0.99:
                new_species = {el: occ / total_occ for el, occ in site.species.items()}
                structure[i] = new_species

    def get_sword_dic(self, child_data: Union[str, Structure], child_label: str | None = None) -> dict[str, Any]:
        """Return the child SWORD label and possible disorder-parent labels."""
        try:
            child = self._parse_input(child_data)
            child.remove_oxidation_states()

            if child_label is None:
                child_label = get_sword_label(
                    child,
                    symprec=self.symprec_child,
                    conventional_struct=True,
                    parser_occ_tolerance=self.parser_occ_tolerance,
                )
            
            self._heal_structure(child)

            structures_to_process = [(child, child_label)]
            
            if self.fill_vacancy and "+" not in child_label:
                vac_results = find_vacancy_ordered(child)
                child_filled = vac_results.get("all_filled_structure")
                if child_filled is not None and isinstance(child_filled, Structure):
                    structures_to_process.append((child_filled, None))

            all_parent_labels = set()

            for struct_target, target_label in structures_to_process:
                if target_label is None:
                    target_label = get_sword_label(
                        struct_target,
                        symprec=self.symprec_child,
                        conventional_struct=True,
                        parser_occ_tolerance=self.parser_occ_tolerance,
                    )
                
                mix_matches = _MIX_PATTERN.findall(target_label)
                
                disordered_els = set()
                for match in mix_matches:
                    for el in match.split('+'):
                        if el != "VAC":
                            disordered_els.add(el)
                
                if disordered_els:
                    mask_sets = [sorted(list(disordered_els))]
                else:
                    elements = sorted([el.symbol for el in struct_target.composition.elements])
                    mask_sets = [
                        comb
                        for k in range(2, len(elements) + 1)
                        for comb in combinations(elements, k)
                    ]

                for mask in mask_sets:
                    try:
                        masked = struct_target.copy()
                        mapping = {el: DummySpecie("X", 0) for el in mask}
                        masked.replace_species(mapping)
                        
                        raw_label = get_sword_label(
                            masked, 
                            symprec=self.symprec_search, 
                            conventional_struct=True, 
                            parser_occ_tolerance=self.parser_occ_tolerance,
                        )

                        mix = "+".join(sorted(mask))
                        fixed_label = self._fix_parent_label(raw_label, mix)
                        all_parent_labels.add(fixed_label)
                    
                    except Exception as e:
                        all_parent_labels.add(f"ERROR_IN_MASK_{'+'.join(mask)}: {str(e)}")

            return {
                "child_label": child_label,
                "parent_labels": list(all_parent_labels),
                "status": "success"
            }

        except Exception as main_e:
            return {
                "child_label": "ERROR",
                "parent_labels": [],
                "status": "failed",
                "error_msg": str(main_e),
                "traceback": traceback.format_exc()
            }

    @staticmethod
    def _label_pool(family_dic: dict[str, Any]) -> set[str]:
        labels = set(family_dic.get("parent_labels", []))
        child_label = family_dic.get("child_label")
        if child_label:
            labels.add(child_label)
        return labels

    def fit(
        self,
        struct1: Union[str, Structure],
        struct2: Union[str, Structure],
        *,
        fill_vacancy: bool | None = None,
    ) -> bool:
        """Return whether two structures share any SWORD family label."""
        if fill_vacancy is None:
            matcher = self
        else:
            matcher = type(self)(
                symprec_child=self.symprec_child,
                symprec_search=self.symprec_search,
                parser_occ_tolerance=self.parser_occ_tolerance,
                fill_vacancy=fill_vacancy,
            )

        dic1 = matcher.get_sword_dic(struct1)
        dic2 = matcher.get_sword_dic(struct2)
        
        if dic1["status"] != "success" or dic2["status"] != "success":
            logger.warning("SWORD Dictionary generation failed for one or both structures.")
            return False
            
        pool1 = self._label_pool(dic1)
        pool2 = self._label_pool(dic2)
        
        # isdisjoint is faster than constructing an intersection set when you just need the boolean
        return not pool1.isdisjoint(pool2)

    def _query_family_dicts(
        self,
        query: Union[str, Structure, pd.Series, dict[str, Any]],
        query_family_cols: tuple[str, ...],
    ) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(query, (str, Structure)):
            return [("generated_query", self.get_sword_dic(query))]

        if isinstance(query, pd.Series):
            query = query.to_dict()

        if isinstance(query, dict) and "child_label" in query and "parent_labels" in query:
            return [("query_family_dic", query)]

        if isinstance(query, dict):
            return [
                (col, query[col])
                for col in query_family_cols
                if col in query and isinstance(query[col], dict)
            ]

        raise TypeError("query must be a Structure, CIF string, dataframe row, or SWORD family dictionary.")

    @staticmethod
    def _default_id_col(ref_df: pd.DataFrame) -> str | None:
        for col in ("CollectionCode", "material_id", "id"):
            if col in ref_df.columns:
                return col
        return None

    @staticmethod
    def _default_ref_family_col(ref_df: pd.DataFrame, ref_family_col: str) -> str:
        if ref_family_col in ref_df.columns:
            return ref_family_col
        if ref_family_col == "SWORD_family_dic" and "SWORD_dic" in ref_df.columns:
            return "SWORD_dic"
        raise ValueError(f"Reference dataframe does not contain family column: {ref_family_col!r}")

    def fit_many(
        self,
        query: Union[str, Structure, pd.Series, dict[str, Any]],
        ref_df: pd.DataFrame,
        *,
        ref_family_col: str = "SWORD_family_dic",
        query_family_cols: tuple[str, ...] = ("SWORD_family_dic", "SWORD_family_dic_vac"),
        id_col: str | None = None,
    ) -> dict[str, Any]:
        """Match one query structure or query row against a reference table."""
        all_query_dicts = self._query_family_dicts(query, query_family_cols)
        query_dicts = [
            (source, family_dic)
            for source, family_dic in all_query_dicts
            if family_dic.get("status") == "success" and family_dic.get("child_label")
        ]

        if not query_dicts:
            first_query_dict = all_query_dicts[0][1] if all_query_dicts else {}
            return {
                "child_label": first_query_dict.get("child_label", "ERROR"),
                "parent_labels": first_query_dict.get("parent_labels", []),
                "matched_ordered_labels": [],
                "matched_ordered_ids": [],
                "matched_disordered_labels": [],
                "matched_disordered_ids": [],
                "matched_source_by_id": {},
                "status": "failed",
                "error_msg": first_query_dict.get(
                    "error_msg",
                    "No successful query SWORD family dictionary was found.",
                ),
                "traceback": first_query_dict.get("traceback", ""),
            }

        query_child_labels = sorted({family_dic["child_label"] for _, family_dic in query_dicts})
        query_parent_labels = sorted(
            {
                label
                for _, family_dic in query_dicts
                for label in family_dic.get("parent_labels", [])
            }
        )
        query_label_pools = {
            source: self._label_pool(family_dic)
            for source, family_dic in query_dicts
        }

        matched_ordered_labels = set()
        matched_ordered_ids = set()
        matched_disordered_labels = set()
        matched_disordered_ids = set()
        matched_source_by_id = {}

        ref_family_col = self._default_ref_family_col(ref_df, ref_family_col)
        id_col = id_col if id_col is not None else self._default_id_col(ref_df)
        ref_ids = ref_df[id_col].values if id_col is not None else ref_df.index.values

        for ref_dict, ref_id in zip(ref_df[ref_family_col].values, ref_ids):
            if not isinstance(ref_dict, dict) or ref_dict.get("status") != "success":
                continue
                
            ref_child_label = ref_dict.get("child_label")
            if not ref_child_label:
                continue
                
            ref_all_labels = self._label_pool(ref_dict)
            matched_sources = [
                source
                for source, query_all_labels in query_label_pools.items()
                if not query_all_labels.isdisjoint(ref_all_labels)
            ]

            if matched_sources:
                if "+" in ref_child_label:
                    matched_disordered_labels.add(ref_child_label)
                    if pd.notna(ref_id):
                        matched_disordered_ids.add(ref_id)
                        matched_source_by_id[ref_id] = sorted(matched_sources)
                else:
                    if ref_child_label not in query_child_labels:
                        matched_ordered_labels.add(ref_child_label)
                        if pd.notna(ref_id):
                            matched_ordered_ids.add(ref_id)
                            matched_source_by_id[ref_id] = sorted(matched_sources)

        return {
            "child_label": query_child_labels[0],
            "parent_labels": query_parent_labels,
            "matched_ordered_labels": list(matched_ordered_labels),
            "matched_ordered_ids": list(matched_ordered_ids),
            "matched_disordered_labels": list(matched_disordered_labels),
            "matched_disordered_ids": list(matched_disordered_ids),
            "matched_source_by_id": matched_source_by_id,
            "status": "success"
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "symprec_child": self.symprec_child,
            "symprec_search": self.symprec_search,
            "parser_occ_tolerance": self.parser_occ_tolerance,
            "fill_vacancy": self.fill_vacancy,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            symprec_child=d.get("symprec_child", 1e-2),
            symprec_search=d.get("symprec_search", 1.0),
            parser_occ_tolerance=d.get("parser_occ_tolerance", d.get("occupancy_tolerance", 10.0)),
            fill_vacancy=d.get("fill_vacancy", True),
        )
