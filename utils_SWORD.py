import os
import warnings
import math
import re
import numpy as np
import pandas as pd
import json
import sys
import itertools
import subprocess
from typing import Any
from typing import Union
from io import StringIO
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifBlock, CifFile
from pymatgen.core.structure import Magmom
from collections import Counter, defaultdict
from itertools import combinations_with_replacement

from itertools import combinations
from pymatgen.core.periodic_table import DummySpecie

radius_df = pd.read_csv('/home/users/yyhuang/ICSD/deduplicate/all_radii.csv')
wyckoff_sets = pd.read_json('/home/users/yyhuang/ICSD/deduplicate/wyckoff_sets.json')

class CifWriter:
    """A customized Pymatgen CIFwrapper to write symmetrized CIF files with wyckoff letter from CIF raw txt."""
    def __init__(
        self,
        cif_txt: str,
        struct: Structure | None = None,
        symprec: float | None = 1e-2,
        write_magmoms: bool = False,
        significant_figures: int = 8,
        angle_tolerance: float = 5,
        occupancy_tolerance: float = 1.0, 
        refine_struct: bool = False,
        conventional_struct: bool = True,
        write_site_properties: bool = False,
        spg_analyzer: SpacegroupAnalyzer = None
    ) -> None:
        """
        Args:
            struct (Structure): structure to write.
            symprec (float): If not none, finds the symmetry of the structure
                and writes the CIF with symmetry information. Passes symprec
                to the SpacegroupAnalyzer. See also refine_struct.
            write_magmoms (bool): If True, will write magCIF file. Incompatible
                with symprec
            significant_figures (int): Specifies precision for formatting of floats.
                Defaults to 8.
            angle_tolerance (float): Angle tolerance for symmetry finding. Passes
                angle_tolerance to the SpacegroupAnalyzer. Used only if symprec
                is not None.
            refine_struct: Used only if symprec is not None. If True, get_refined_structure
                is invoked to convert input structure from primitive to conventional.
            write_site_properties (bool): Whether to write the Structure.site_properties
                to the CIF as _atom_site_{property name}. Defaults to False.
        """

        if write_magmoms and symprec is not None:
            warnings.warn(
                "Magnetic symmetry cannot currently be detected by pymatgen, disabling symmetry detection.",
                stacklevel=2,
            )
            symprec = None

        if struct is None:
            if any(k in cif_txt for k in ("_atom_site", "_cell_length", "data_")):
                cif_str = StringIO(cif_txt)
                parser = CifParser(cif_str, occupancy_tolerance=occupancy_tolerance)
                struct = parser.parse_structures(primitive=False)[0]
            else:
                struct = Structure.from_str(cif_txt, fmt="poscar")

        blocks: dict[str, Any] = {}
        spacegroup: tuple[str, int] = ("P 1", 1)
        if symprec is not None:
            if spg_analyzer is not None:
                spg_analyzer = spg_analyzer
            else:
                spg_analyzer = SpacegroupAnalyzer(struct, symprec, angle_tolerance=angle_tolerance)
            spacegroup = (
                spg_analyzer.get_space_group_symbol(),
                spg_analyzer.get_space_group_number(),
            )

            if refine_struct:
                # Need the refined structure when using symprec. This converts
                # primitive to conventional structures, the standard for CIF, 
                # with atoms moved to the expected symmetry positions.
                struct = spg_analyzer.get_refined_structure()
            if conventional_struct:
                # Get a structure with a conventional cell according to certain standards.
                # This is not necessarily the same as the standard settings within the 
                # International Tables of Crystallography, for which get_refined_structure 
                # should be used instead.
                struct = spg_analyzer.get_conventional_standard_structure()

        lattice = struct.lattice
        comp = struct.composition
        no_oxi_comp = comp.element_composition
        format_str: str = f"{{:.{significant_figures}f}}"
        blocks["_symmetry_space_group_name_H-M"] = spacegroup[0]
        for cell_attr in ("a", "b", "c"):
            blocks[f"_cell_length_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
        for cell_attr in ("alpha", "beta", "gamma"):
            blocks[f"_cell_angle_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
        blocks["_symmetry_Int_Tables_number"] = spacegroup[1]
        blocks["_chemical_formula_structural"] = no_oxi_comp.reduced_formula
        blocks["_chemical_formula_sum"] = no_oxi_comp.formula
        blocks["_cell_volume"] = format_str.format(lattice.volume)

        _, fu = no_oxi_comp.get_reduced_composition_and_factor()
        blocks["_cell_formula_units_Z"] = str(int(fu))

        if symprec is None:
            blocks["_symmetry_equiv_pos_site_id"] = ["1"]
            blocks["_symmetry_equiv_pos_as_xyz"] = ["x, y, z"]

        else:
            spg_analyzer = SpacegroupAnalyzer(struct, symprec)
            symm_ops: list[SymmOp] = []
            for op in spg_analyzer.get_symmetry_operations():
                v = op.translation_vector
                symm_ops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

            ops = [op.as_xyz_str() for op in symm_ops]
            blocks["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
            blocks["_symmetry_equiv_pos_as_xyz"] = ops

        loops: list[list[str]] = [
            ["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"],
        ]

        try:
            symbol_to_oxi_num = {str(el): float(el.oxi_state or 0) for el in sorted(comp.elements)}
            blocks["_atom_type_symbol"] = list(symbol_to_oxi_num)
            blocks["_atom_type_oxidation_number"] = symbol_to_oxi_num.values()
            loops.append(["_atom_type_symbol", "_atom_type_oxidation_number"])
        except (TypeError, AttributeError):
            symbol_to_oxi_num = {el.symbol: 0 for el in sorted(comp.elements)}

        atom_site_type_symbol = []
        atom_site_symmetry_multiplicity = []
        atom_site_fract_x = []
        atom_site_fract_y = []
        atom_site_fract_z = []
        atom_site_label = []
        atom_site_occupancy = []
        atom_site_moment_label = []
        atom_site_moment_crystalaxis_x = []
        atom_site_moment_crystalaxis_y = []
        atom_site_moment_crystalaxis_z = []
        atom_site_properties: dict[str, list] = defaultdict(list)
        wyckoffs: list[str] = []  #####################
        count = 0
        if symprec is None:
            for site in struct:
                for sp, occu in sorted(site.species.items()):
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append("1")
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    atom_site_occupancy.append(str(occu))
                    site_label = f"{sp.symbol}{count}"

                    if "magmom" in site.properties:
                        mag = site.properties["magmom"]
                    elif getattr(sp, "spin", None) is not None:
                        mag = sp.spin
                    else:
                        # Use site label if available for regular sites
                        site_label = site.label if site.label != site.species_string else site_label
                        mag = 0

                    atom_site_label.append(site_label)

                    magmom = Magmom(mag)
                    if write_magmoms and abs(magmom) > 0:
                        moment = Magmom.get_moment_relative_to_crystal_axes(magmom, lattice)
                        atom_site_moment_label.append(f"{sp.symbol}{count}")
                        atom_site_moment_crystalaxis_x.append(format_str.format(moment[0]))
                        atom_site_moment_crystalaxis_y.append(format_str.format(moment[1]))
                        atom_site_moment_crystalaxis_z.append(format_str.format(moment[2]))

                    if write_site_properties:
                        for key, val in site.properties.items():
                            atom_site_properties[key].append(format_str.format(val))

                    count += 1

        else:
            # The following just presents a deterministic ordering
            symm_struct = spg_analyzer.get_symmetrized_structure() #############
            dataset = spg_analyzer.get_symmetry_dataset()###############
            wyck_all = dataset.wyckoffs ###############
            unique_sites = [
                (
                    min(sites, key=lambda site: tuple(abs(x) for x in site.frac_coords)),
                    len(sites),
                    wyck_all[indices[0]],#############
                )
                for sites, indices in zip(symm_struct.equivalent_sites,symm_struct.equivalent_indices,)  # type: ignore[reportPossiblyUnboundVariable]
            ]#############
            for site, mult, wyck in sorted( ##########
                unique_sites,
                key=lambda t: (
                    t[0].species.average_electroneg,
                    -t[1],
                    t[0].a,
                    t[0].b,
                    t[0].c,
                ),
            ):
                for sp, occu in site.species.items():
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append(f"{mult}")
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    site_label = site.label if site.label != site.species_string else f"{sp.symbol}{count}"
                    atom_site_label.append(site_label)
                    atom_site_occupancy.append(str(occu))
                    wyckoffs.append(wyck) ##########
                    count += 1

        if len(set(atom_site_label)) != len(atom_site_label):
            warnings.warn(
                "Site labels are not unique, which is not compliant with the CIF spec "
                "(https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Iatom_site_label.html):"
                f"`{atom_site_label}`.",
                stacklevel=2,
            )

        blocks["_atom_site_type_symbol"] = atom_site_type_symbol
        blocks["_atom_site_label"] = atom_site_label
        blocks["_atom_site_symmetry_multiplicity"] = atom_site_symmetry_multiplicity
        if wyckoffs:
            blocks["_atom_site_Wyckoff_symbol"] = wyckoffs ###########
        blocks["_atom_site_fract_x"] = atom_site_fract_x
        blocks["_atom_site_fract_y"] = atom_site_fract_y
        blocks["_atom_site_fract_z"] = atom_site_fract_z
        blocks["_atom_site_occupancy"] = atom_site_occupancy
        loop_labels = [
            "_atom_site_type_symbol",
            "_atom_site_label",
            "_atom_site_symmetry_multiplicity",]
        if wyckoffs:
            loop_labels.append("_atom_site_Wyckoff_symbol") ##########
        loop_labels += [
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]

        if write_site_properties:
            for key, vals in atom_site_properties.items():
                blocks[f"_atom_site_{key}"] = vals
                loop_labels += [f"_atom_site_{key}"]
        loops.append(loop_labels)

        if write_magmoms:
            blocks["_atom_site_moment_label"] = atom_site_moment_label
            blocks["_atom_site_moment_crystalaxis_x"] = atom_site_moment_crystalaxis_x
            blocks["_atom_site_moment_crystalaxis_y"] = atom_site_moment_crystalaxis_y
            blocks["_atom_site_moment_crystalaxis_z"] = atom_site_moment_crystalaxis_z
            loops.append(
                [
                    "_atom_site_moment_label",
                    "_atom_site_moment_crystalaxis_x",
                    "_atom_site_moment_crystalaxis_y",
                    "_atom_site_moment_crystalaxis_z",
                ]
            )
        dct = {comp.reduced_formula: CifBlock(blocks, loops, comp.reduced_formula)}
        self._cf = CifFile(dct)

def is_symmetrized_CIF(raw_txt, sym_txt):
    if not any(k in raw_txt for k in ("_atom_site", "_cell_length", "data_")):
        return False

    def count_sites(txt):
        lines = txt.splitlines()
        count, in_block = 0, False
        for ln in lines:
            if ln.strip().startswith("loop_"):
                in_block = False
            if "_atom_site_Wyckoff_symbol" in ln:
                in_block = True
                continue
            if in_block:
                s = ln.strip()
                if s.startswith("_"):
                    in_block = False
                elif s:
                    count += 1
        return count

    if count_sites(raw_txt) != count_sites(sym_txt):
        return False
    
    def get_it(txt):
        for key in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
            m = re.search(rf"{key}\s+['\"]?(\S+?)['\"]?(?:\s|$)", txt)
            if m:
                return m.group(1)
        return None

    raw_it = get_it(raw_txt)
    sym_it = get_it(sym_txt)
    if raw_it is not None and sym_it is not None and raw_it != sym_it:
        warnings.warn(
            "It appears that a symmetrized CIF is provided, but the "
            "_space_group_IT_number recorded in the original file "
            f"({raw_it}) is inconsistent with the space group obtained "
            f"after symmetrization ({sym_it}) using spglib. Please double-check "
            f"the structure and its symmetry settings. Using the CIF symmetrized (space group {sym_it}) by spglib."
        )
        return False

    if "_atom_site_Wyckoff_symbol" not in raw_txt:
        return False

    return True

def find_symprec(raw_txt, *, n=2, symprec=1e-2, angle_tolerance=5.0, occupancy_tolerance:float = 1.0, merge_tolerance:float = 0.01):
    """
    Find symprec that yields a valid symmetry analysis matching the CIF space group. Cleanup of too closed sites below tolerance.
    """
    # retrieve target space group from CIF
    m = re.search(r"_(?:space_group_IT_number|symmetry_Int_Tables_number)\s+['\"]?(\S+?)['\"]?(?:\s|$)", raw_txt, re.IGNORECASE)
    if not m:
        raise ValueError("No _space_group_IT_number found in CIF text.")
    target_sg = int(m.group(1))

    #merge_triggers = ("Incorrect stoichiometry",)
    #with warnings.catch_warnings(record=True) as warn_list:
    #        warnings.simplefilter("always")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_structure = CifParser(StringIO(raw_txt), occupancy_tolerance = occupancy_tolerance).parse_structures(primitive=False)[0]

    merged_base = False
    # for w in warn_list:
    #     if any(k in str(w.message) for k in merge_triggers):
    #         warnings.warn((f"[Warning Triggered] {w.message}"))
    #         base_structure = base_structure.merge_sites(mode="delete", tol = merge_tolerance)
    #         merged_base = True
    #         break

    candidates = [symprec]
    for i in range(1, n + 1):
        candidates.append(symprec / (10**i))
        candidates.append(symprec * (10**i))
        
    parsed_ok = False
    best_diff = None
    best_match = None # (sp, merged, struct, sga, sgnum)
    for sp in candidates:
        struct = base_structure.copy()
        merged = merged_base  # initial state

        try:
            sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
            sgnum = sga.get_space_group_number()
            parsed_ok = True
            if target_sg == 1 or int(sgnum) == target_sg:
                return sp, merged, struct, sga
            
            if parsed_ok and sgnum is not None: #actually this condition is unnecessary
                diff = abs(int(sgnum) - target_sg)

            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_match = (sp, merged, struct, sga, sgnum)
        except Exception:
            sgnum = None  # fall through to merge retry
        
        # if not merged:
        #     # retry after merging close sites
        #     struct = struct.merge_sites(mode="delete", tol = merge_tolerance) 
        #     merged = True
        #     try:
        #         sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
        #         sgnum = sga.get_space_group_number()
        #         parsed_ok = True
        #         if target_sg == "1" or str(sgnum) == target_sg:
        #             return sp, merged, struct, sga
        #     except Exception:
        #         continue

    if not parsed_ok:
        raise RuntimeError(f"No symprec setting allowed SpacegroupAnalyzer to parse the structure within {2 * (1 + 2*n)} attempts.")
    if best_match is not None:
        sp_best, merged, struct, sga, sgnum_best = best_match
        warnings.warn(f"No symprec setting could reproduced the original space group {target_sg} within {2 * (1 + 2*n)} iterations;"
                      f"returning closest match {sgnum_best} at symprec={sp_best}.")
        return sp_best, merged, struct, sga
    
def clean_num(token):
    if token is None:
        return None
    if isinstance(token, (int, float)):
        return float(token)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    token = re.sub(r"\(.*\)", "", token)
    try:
        return float(token)
    except Exception:
        return None

def symm_orbits_df(
    structure: Structure,
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5,
    id_key: str = "stable_id",
) -> pd.DataFrame:
    """
    Summarize symmetry-equivalent site orbits (with Wyckoff letters) as a DataFrame.

    The returned table is meant for debugging how sites are grouped under symmetry.
    Per-member element/label/id information is taken from the *input* structure via
    equivalent_indices (i.e., it is not affected by CIF re-writing/re-parsing).
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    symm_struct = sga.get_symmetrized_structure()
    dataset = sga.get_symmetry_dataset()

    rows: list[dict[str, Any]] = []
    for sites, idxs in zip(symm_struct.equivalent_sites, symm_struct.equivalent_indices):
        idxs = list(idxs)
        rep_idx = idxs[0]
        rep_site = structure[rep_idx]

        member_sites = [structure[i] for i in idxs]
        member_ids = [s.properties.get(id_key) for s in member_sites]
        member_elements = [
            tuple(sorted(el.symbol for el in s.species.elements)) for s in member_sites
        ]

        rows.append(
            {
                "wyckoff_symbol": dataset.wyckoffs[rep_idx],
                "multiplicity": len(idxs),
                "rep_symm_frac": list(sites[0].frac_coords),
                "rep_species": rep_site.species_string,
                "rep_species_detail": str(rep_site.species),
                "rep_elements": tuple(sorted(el.symbol for el in rep_site.species.elements)),
                "member_indices": idxs,
                "member_ids": member_ids,
                "member_labels": [getattr(s, "label", None) for s in member_sites],
                "member_species": [s.species_string for s in member_sites],
                "member_elements": member_elements,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values(["wyckoff_symbol", "multiplicity"], ascending=[True, False])
            .reset_index(drop=True)
        )
    return df

class StructureEntry:
    """
    ICSD CIF wrapper. Reads a CIF block, preserves the raw data, and optionally
    fills missing crystallographic info using pymatgen + spglib.
    """
    def __init__(self, cif_text, collection_code=None, meta=None):
        self.CollectionCode = collection_code
        parser = CifParser(StringIO(cif_text))
        cif_dict = parser.as_dict()
        block_name = next(iter(cif_dict))
        block = cif_dict[block_name]

        payload = self._payload_from_block(block, cif_dict, cif_text)
        self._apply_payload(payload, meta)

    @staticmethod
    def _payload_from_block(block, cif_dict, cif_text):
        raw_sg = block.get("_space_group_IT_number") or block.get("_symmetry_Int_Tables_number")
        sym_ops_raw = block.get("_space_group_symop_operation_xyz") or block.get("_symmetry_equiv_pos_as_xyz")
        try:
            sg_number = int(re.sub(r"\(.*\)", "", raw_sg))
        except Exception:
            sg_number = (
                block.get("_space_group_IT_number") or block.get("_symmetry_Int_Tables_number")
            )

        payload = {
            "cif_text": cif_text,
            "cif_dict": cif_dict,
            "space_group_number": sg_number,
            "lattice": (
                clean_num(block.get("_cell_length_a")),
                clean_num(block.get("_cell_length_b")),
                clean_num(block.get("_cell_length_c")),
            ),
            "labels": list(block.get("_atom_site_label", [])),
            "type_symbols": list(block.get("_atom_site_type_symbol", [])),
            "coords": list(
                zip(
                    [clean_num(x) for x in block.get("_atom_site_fract_x", [])],
                    [clean_num(y) for y in block.get("_atom_site_fract_y", [])],
                    [clean_num(z) for z in block.get("_atom_site_fract_z", [])],
                )
            ),
            "occupancies": [clean_num(x) for x in block.get("_atom_site_occupancy", [])],
            "multiplicities": [int(m) for m in block.get("_atom_site_symmetry_multiplicity", [])],
            "wyckoffs": list(block.get("_atom_site_Wyckoff_symbol", [])),
            "sym_ops": [SymmOp.from_xyz_str(op) for op in sym_ops_raw],
        }
        return payload

    def _apply_payload(self, payload, meta):
        self.meta = meta or {}
        self.cif_str = payload["cif_text"]
        self.cif_dict = payload["cif_dict"]
        self.spg_num = int(payload["space_group_number"])
        self.lattice = payload["lattice"]

        self.labels = payload["labels"]
        self.type_symbols = payload["type_symbols"]
        self.points = [tuple(pt) for pt in payload["coords"]]
        self.occ = payload["occupancies"]
        self.mults = payload["multiplicities"]
        self.wyckoffs = payload["wyckoffs"]
        self.sym_ops = [op if isinstance(op, SymmOp) else SymmOp.from_xyz_str(op) for op in payload["sym_ops"]]

        if not hasattr(self, "read_by"):
            self.read_by = "raw"

        if not hasattr(self, "source"):
            self.source = "ICSD"

        if not hasattr(self, "is_merge"):
            self.is_merge = False

        self.records = []
        for i, label in enumerate(self.labels):
            self.records.append(
                {
                    "label": label,
                    "type_symbol": self.type_symbols[i],
                    "multiplicity": self.mults[i],
                    "wyckoff_symbol": self.wyckoffs[i],
                    "coordinate": list(self.points[i]),
                    "occupancy": self.occ[i],
                }
            )
        self.records = sorted(self.records, key=lambda x: x["label"])
        self.df = pd.DataFrame(self.records)

    @classmethod
    def from_txt(cls, raw_txt, *, code = None, meta=None, symprec=1e-2, angle_tolerance=5, occupancy_tolerance: float = 1.0, source: str = 'None', conventional_struct: bool = True, refine_struct: bool = False):
        #construct a ICSD-style symmetrized CIF_txt from raw CIF or POSCAR
        try: # find_symprec need to be  modified to adjust POSCAR
            symprec, is_merge, struct, sga = find_symprec(raw_txt, n = 2, symprec=symprec, angle_tolerance=5, occupancy_tolerance = occupancy_tolerance)
        except Exception as e:
            warnings.warn(
                f"[StructureEntry.from_txt] find_symprec failed: falling back to use default symprec."
                f"symprec={symprec}, angle_tol={angle_tolerance}. "
                f"Error: {e}")
            symprec, is_merge, struct, sga = symprec, False, None, None
        sym_txt = str(CifWriter(raw_txt, symprec=symprec, angle_tolerance=angle_tolerance, occupancy_tolerance=occupancy_tolerance, struct=struct, spg_analyzer=sga, conventional_struct=conventional_struct, refine_struct=refine_struct)._cf)
        # Check completeness of input to determine whether to parse raw or symmetrized file
        #use_sym = not is_symmetrized_CIF(raw_txt, sym_txt)
        #chosed_txt = sym_txt if use_sym else raw_txt  
        chosed_txt = sym_txt
        parser = CifParser(StringIO(chosed_txt))
        cif_dict = parser.as_dict()
        block_name = next(iter(cif_dict))
        block = cif_dict[block_name]

        payload = cls._payload_from_block(block, cif_dict, chosed_txt)

        entry = cls.__new__(cls)
        entry.CollectionCode = code
        entry.is_merge = is_merge
        #entry.read_by = "pmg" if use_sym else "raw"
        entry.read_by = "pmg"
        entry.source = source
        entry._apply_payload(payload, meta)
        return entry

    @classmethod
    def from_collection_code(cls, collection_code, ICSD_df: pd.DataFrame = None, meta=None):
        row = ICSD_df[ICSD_df["CollectionCode"] == collection_code].iloc[0]
        cif_text = row["cif"]
        if meta == "all":
            meta = row.to_dict()
        return cls(cif_text, meta=meta, collection_code=collection_code)
    
def periodic_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff_pbc = diff - np.round(diff)
    return np.linalg.norm(diff_pbc)

def get_equiv_positions(point, sym_ops, frac_tolerance=1e-4):
    """Computes the set of unique equivalent positions for a given fractional coordinate 
    using a list of symmetry operations {sym_ops} from CIF. The generated raw positions 
    will be clustered when periodic Euclidean distance tolerance <= frac_tolerance and 
    floating-error derived duplicated positions will be deduplicated.
    returning the coordinates as a sorted tuple."""
    p0 = np.asarray(point, dtype=float) % 1.0
    orb_raw = []
    for op in sym_ops:
        if hasattr(op, "operate"):
            p = op.operate(p0)
        else:
            p = op(p0)
        orb_raw.append(p % 1.0) 
    # display(orb_raw)

    clusters = [] 
    for p_new in orb_raw:
        found = False
        for rep, points_in_cluster in clusters:
            #print(f"p_new{p_new}, rep: {rep}, distance:{periodic_dist(p_new, rep)}")
            if periodic_dist(p_new, rep) <= frac_tolerance*2:
                points_in_cluster.append(p_new)
                found = True
                break

        if not found:
            clusters.append([p_new.copy(), [p_new.copy()]])

    # for i, cluster in enumerate(clusters):
    #     rep = cluster[0]
    #     print(f"Cluster {i+1}: Representative: {np.round(rep, 6)}")
    # print(f"total clusters: {len(clusters)}")

    final_coords = []
    ndigits = max(0, -int(math.floor(math.log10(frac_tolerance))))
    for rep, _ in clusters: 
        coord_tuple = tuple(
            float(x)
            for x in np.round(
                (np.round((rep % 1.0) / frac_tolerance) * frac_tolerance) % 1.0,
                ndigits,))

        if coord_tuple not in final_coords:
            final_coords.append(coord_tuple)
                       
    return tuple(sorted(final_coords))

def compute_disorder(entry, disordered_list, total_sites = 0, verbose = False):
    """
    For disorder structures, quantifies the degree of mixing(DOM) in a structure. The function utilizes 
    Shannon Entropy to calculate the mixing factor (site_mixing) for each disordered site. 

    The Degree of Mixing has sign delta: delta = (XA - XZ) / |XA - XZ|
    To establish a directional DOM, the site with the highest contribution to total mixing value is chosen as the representative. 
    A sign factor (delta) is then derived from the two most extreme components (minimum XA and maximum XZ occupancies) of this representative site.

    Returns the fraction of disordered sites and the overall degree of mixing (DOM) for the entire structure.
    """
    from collections import defaultdict
    all_labels = entry.labels
    label2mult = dict(zip(all_labels, entry.mults))
    label2type = dict(zip(all_labels, entry.type_symbols))
    label2occ = dict(zip(all_labels, entry.occ))

    disordered_sites = 0
    total_mixing = 0.0
    rep_site_mixing = float("-inf")
    rep_merge_occ = None

    for sites in disordered_list:
        non_vac_labels = [lab for lab in sites if "VAC" not in lab]
        non_vac_type_symbols = [label2type[lab] for lab in non_vac_labels if label2occ[lab]>0]
        occs = [label2occ[lab] for lab in non_vac_labels if label2occ[lab]>0]

        merged_occ = defaultdict(float) #merge the occ, because some situation will include same valence element in disorder_list
        for s, o in zip(non_vac_type_symbols, occs):
            merged_occ[s] += o

        occs = list(merged_occ.values()) #should check if merged_occ > 1.0,可能需要增加一个空值occ溢出的参数

        if any("VAC" in lab for lab in sites):
            occ_vac = 1.0 - sum(occs)
            occs.append(occ_vac)
            merged_occ["VAC"] = occ_vac
        if len(occs) > 1:
            site_mixing = -sum(p * math.log(p) for p in occs)
            site_mixing = site_mixing / math.log(len(occs))
        else:
            site_mixing = 0.0

        multiplicity = label2mult[non_vac_labels[0]]

        if site_mixing * multiplicity > rep_site_mixing * multiplicity:
            rep_site_mixing = site_mixing
            rep_merge_occ = merged_occ

        disordered_sites += multiplicity
        total_mixing += site_mixing * multiplicity
        if verbose == True:
            print(f"the site is occupied with: {dict(merged_occ)}\n"
                    f"site mixing factor: {site_mixing}\n"
                    f"number of sites: {multiplicity}")
    
    if len(rep_merge_occ) > 1:
        kmin = min(rep_merge_occ, key=rep_merge_occ.get)
        kmax = max(rep_merge_occ, key=rep_merge_occ.get)
        a, b = sorted((kmin, kmax))
        XA = rep_merge_occ[a]
        XZ = rep_merge_occ[b]
        if XA == XZ:
            delta = 1
        else:
            delta = (XA - XZ) / abs(XA - XZ)
        if verbose == True:
            print(kmin, kmax, XA, XZ, delta)
    else:
        delta = 0
        if verbose == True:
            print(delta)

    #ordered_sites = sum(label2mult[lab] for lab in all_labels_list if lab not in all_disordered_labels)
    #total_sites = disordered_sites + ordered_sites
    if verbose == True:
            print(f"disordered_sites: {disordered_sites}, total_sites: {total_sites}, total_mixing: {total_mixing}")
    fraction_of_disordered_sites = round(disordered_sites / total_sites, 4)
    degree_of_mixing = round(delta* total_mixing / disordered_sites, 4)

    return fraction_of_disordered_sites, degree_of_mixing

#------mapping wyckoff_sets into a nested dictionary with spacegroup number and coset number indexed each Transformed WP-------------
mapping_by_sg = {}
for sg in wyckoff_sets.index:
    no_list = wyckoff_sets.at[sg, "No."]                     # e.g. [1,2,3,4]
    wp_str_list = wyckoff_sets.at[sg, "Transformed WP"]      # e.g. ['a b c d e f g h i j', ...]
    inner_map = {}
    for no, wp_str in zip(no_list, wp_str_list):
        letters = wp_str.split()                             
        letter_map = {idx: ch for idx, ch in enumerate(letters)}
        inner_map[no] = letter_map  
    mapping_by_sg[sg] = inner_map

_KEY_PAT = re.compile(r"^([A-Za-z]+)(\d*)$")
_SWORD_DIR = os.path.dirname(__file__)

# ---- Hall-only table loader (once) ----
_HALL_ONLY_JSON = os.path.join(_SWORD_DIR, "hall_letter_maps_by_sg.json")
with open(_HALL_ONLY_JSON, "r") as f:
    _HALL_ONLY_RAW = json.load(f)

def _get_hall_maps_for_sg(sg_num):
    """
    Return {hall_no(str): perm_ref_to_hall(dict)} for one SG.
    Supports:
    1) old format: { "12": { "63": {...}, ... }, ... }
    2) tagged format: { "by_sg": { "12": { "halls": { "63": {"perm_ref_to_hall": {...}}, ... }}}}
    """
    sg_key = str(int(sg_num))

    if "by_sg" in _HALL_ONLY_RAW:
        block = _HALL_ONLY_RAW["by_sg"].get(sg_key, {})
        halls = block.get("halls", {})
        out = {}
        for h, rec in halls.items():
            perm = rec.get("perm_ref_to_hall", rec) if isinstance(rec, dict) else rec
            out[str(h)] = {str(k): str(v) for k, v in perm.items()}
        return out

    block = _HALL_ONLY_RAW.get(sg_key, {})
    return {str(h): {str(k): str(v) for k, v in p.items()} for h, p in block.items()}

def _apply_letter_perm_to_sequence_map(sequence_map, letter_perm):
    out = {}
    for k, v in sequence_map.items():
        m = _KEY_PAT.fullmatch(str(k).strip())
        if not m:
            raise ValueError(f"Invalid Wyckoff key format: {k!r}")
        letter, suffix = m.group(1), (m.group(2) or "")
        out[f"{letter_perm[letter]}{suffix}"] = v
    return out

def _hall_candidates_and_conventional(sequence_map, sg_num):
    """
    Return:
      hall_candidates: list of dicts (for debug/inspection)
      seq_map_conv: sequence_map mapped to selected Hall branch
      conv_branch: selected candidate dict
    """
    hall_maps = _get_hall_maps_for_sg(sg_num)
    if not hall_maps:
        seq_sorted = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
        return [], seq_sorted, {"hall_no": None, "wyck": "_".join(seq_sorted.keys()), "elem": "_".join(seq_sorted.values())}

    cands = []
    for hall_no, perm_ref_to_hall in sorted(hall_maps.items(), key=lambda kv: int(kv[0])):
        # input assumed in hall_no; map back to reference/conventional with inverse
        perm_hall_to_ref = {v: k for k, v in perm_ref_to_hall.items()}
        transformed = _apply_letter_perm_to_sequence_map(sequence_map, perm_hall_to_ref)
        transformed = dict(sorted(transformed.items(), key=lambda kv: (kv[1], kv[0])))

        cands.append({
            "hall_no": int(hall_no),
            "wyck": "_".join(transformed.keys()),
            "elem": "_".join(transformed.values()),
            "sequence_map_ref": transformed,
        })

    # dedup by (wyck, elem)
    uniq = {}
    for c in cands:
        uniq.setdefault((c["wyck"], c["elem"]), c)
    hall_candidates = list(uniq.values())

    # choose Hall-stage lexicographic minimum, same tie-break style as WP canonicalization
    conv_branch = min(hall_candidates, key=lambda x: (x["wyck"], x["elem"]))
    return hall_candidates, conv_branch["sequence_map_ref"], conv_branch

def _enumerate_wp_candidates_legacy(sequence_map, sg_num):
    """
    Legacy WP candidates from mapping_by_sg (same rule as old canonical).
    """
    wp_mappings = mapping_by_sg.get(sg_num, {})
    if not wp_mappings:
        t = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
        return [("_".join(t.keys()), "_".join(t.values()), t)]

    base_no = min(wp_mappings)
    base_idx_to_letter = wp_mappings[base_no]
    base_letter_to_idx = {ch: idx for idx, ch in base_idx_to_letter.items()}

    candidates = []
    for _, idx_to_letter in wp_mappings.items():
        transformed = {}
        ok = True
        for k, v in sequence_map.items():
            m = _KEY_PAT.fullmatch(str(k).strip())
            if not m:
                ok = False
                break
            ch, suf = m.group(1), (m.group(2) or "")
            idx = base_letter_to_idx.get(ch)
            if idx is None or idx not in idx_to_letter:
                ok = False
                break
            transformed[f"{idx_to_letter[idx]}{suf}"] = v
        if not ok:
            continue

        transformed = dict(sorted(transformed.items(), key=lambda kv: (kv[1], kv[0])))
        candidates.append(("_".join(transformed.keys()), "_".join(transformed.values()), transformed))

    # unique + sorted
    uniq = {}
    for w, e, t in candidates:
        uniq[(w, e)] = t
    return [(w, e, uniq[(w, e)]) for (w, e) in sorted(uniq.keys())]

def _canonicalize_sequence_map_with_hall(sequence_map, sg_num, return_trace=False):
    """
    Hall-stage normalization + legacy WP canonicalization.
    Shared by CIF/native SWORD path and pyxtal_dict path.
    """
    hall_candidates, seq_map_conv, conv_branch = _hall_candidates_and_conventional(sequence_map, sg_num)

    # legacy canonical (3 args)
    wyckoff_set_std, element_seq, seq_std = get_canonical_wyckoff_sets(
        seq_map_conv, mapping_by_sg, sg_num
    )

    if not return_trace:
        return wyckoff_set_std, element_seq, seq_std

    wp_candidates = _enumerate_wp_candidates_legacy(seq_map_conv, sg_num)
    return wyckoff_set_std, element_seq, seq_std, {
        "hall_candidates": hall_candidates,
        "conventional_branch": conv_branch,
        "wp_candidates_from_conventional": wp_candidates,
    }

# backward-compatible alias
def _canonicalize_pyxtal_sequence_map(sequence_map, sg_num, return_trace=False):
    return _canonicalize_sequence_map_with_hall(sequence_map, sg_num, return_trace=return_trace)

def get_canonical_wyckoff_sets(sequence_map, mapping_by_sg, space_group_number):
    """
    Canonicalize (wyckoff_part -> elem_part) by enumerating all Transformed WP mappings,
    re-sorting each transformed map by (value, key), and taking lexicographically minimal
    (wyckoff_seq, element_seq).
    """
    wp_mappings = mapping_by_sg.get(space_group_number, {})
    # fallback: no mapping table
    if not wp_mappings:
        seq_sorted = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
        return "_".join(seq_sorted.keys()), "_".join(seq_sorted.values()), seq_sorted

    base_no = min(wp_mappings)
    base_idx_to_letter = wp_mappings[base_no]
    base_letter_to_idx = {ch: idx for idx, ch in base_idx_to_letter.items()}

    def transform_key(k, idx_to_letter):
        m = re.fullmatch(r"([A-Za-z]+)(\d*)", k)
        if not m:
            return None
        ch, suf = m.group(1), m.group(2) or ""
        idx = base_letter_to_idx.get(ch)
        if idx is None:
            return None
        new_ch = idx_to_letter.get(idx)
        if new_ch is None:
            return None
        return f"{new_ch}{suf}"

    candidates = []
    for _, idx_to_letter in wp_mappings.items():
        transformed = {}
        ok = True
        for k, v in sequence_map.items():
            nk = transform_key(k, idx_to_letter)
            if nk is None:
                ok = False
                break
            transformed[nk] = v
        if not ok:
            continue

        transformed_sorted = dict(sorted(transformed.items(), key=lambda kv: (kv[1], kv[0])))
        wyck_seq = "_".join(transformed_sorted.keys())
        elem_seq = "_".join(transformed_sorted.values())
        candidates.append((wyck_seq, elem_seq, transformed_sorted))

    if not candidates:
        seq_sorted = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
        return "_".join(seq_sorted.keys()), "_".join(seq_sorted.values()), seq_sorted

    wyck_seq, elem_seq, seq_std = min(candidates, key=lambda t: (t[0], t[1]))
    return wyck_seq, elem_seq, seq_std


def mean_site_radius(species_list):
    """
    Compute mean radius for a crystallographic site occupied by species_list.
    species_list: list of species entries. Each entry may be:
       - string: e.g. "Fe2+", "O2-", "C", "Fe3"
    radius_df: pandas.DataFrame with columns ['symbol','charge','ionic radius','empirical']
    Returns: float (mean radius)
    Raises: ValueError if any element has no data in radius_df.
    """
    radii = []
    if not species_list:
        raise ValueError("species_list is empty")

    for entry in species_list:
        symbol = None
        ox = None

        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            symbol = str(entry[0])
            ox = entry[1] if len(entry) > 1 else None

        elif isinstance(entry, str):
            m = re.match(r"^([A-Z][a-z]?)(.*)$", entry.strip())
            if not m:
                raise ValueError(f"Cannot parse species string: {entry!r}")
            symbol = m.group(1)
            rest = m.group(2)
            mnum = re.search(r"([+-]?\d+(\.\d+)?)", rest)
            ox = float(mnum.group(1)) if mnum else None

        else:
            if hasattr(entry, "element"):
                element_obj = entry.element
                symbol = getattr(element_obj, "symbol", str(element_obj))
                ox = getattr(entry, "oxi_state", None)
                if ox is None:
                    ox = getattr(entry, "oxidation_state", None)
                if ox is None:
                    ox = getattr(entry, "oxid_state", None)
            else:
                s = str(entry)
                m = re.match(r"^([A-Z][a-z]?)(.*)$", s.strip())
                if m:
                    symbol = m.group(1)
                    rest = m.group(2)
                    mnum = re.search(r"([+-]?\d+(\.\d+)?)", rest)
                    ox = float(mnum.group(1)) if mnum else None
                else:
                    raise ValueError(f"Unsupported species entry type: {entry!r}")
        # ---- round oxidation if present ----
        ox_int = None
        if ox is not None:
            try:
                ox_int = int(round(float(ox)))
            except Exception:
                ox_int = None
        # ---- lookup: ionic radius first, then empirical ----
        radius_val = None
        if ox_int is not None:
            rows = radius_df[(radius_df['symbol'] == symbol) & (radius_df['charge'] == ox_int)]
            if not rows.empty:
                radius_val = float(rows['ionic radius'].min())

        if radius_val is None:
            rows_emp = radius_df[radius_df['symbol'] == symbol]
            if not rows_emp.empty and 'empirical' in rows_emp.columns:
                radius_val = float(rows_emp['empirical'].min())

        if radius_val is None:
            raise ValueError(f"No radius data for element '{symbol}' (entry {entry!r}) in radius_df")

        radii.append(radius_val)

    return float(np.mean(radii))

def intersect_orb(entry, site_tolerance = 1e-4, occ_tolerance = 1.0):
    """
    Checks for positional disorder in a crystallographic entry by examining atom-to-atom 
    distances (not site-to-site wyckoff position distance) against a chemically-derived radius threshold
    derived from function mean_site_radius. The site species intersect_orb, 
    and flags any sites found to be closer than this threshold (or 1.0 Å, whichever is greater).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crystal = Structure.from_str(entry.cif_str, fmt="cif",
                                        site_tolerance = site_tolerance,
                                        occupancy_tolerance = occ_tolerance
                                        )
        all_site_labels = [s.label for s in crystal]
        all_site_species = [list(s.species.keys()) for s in crystal]

        site_keys = [tuple(sorted((
                    sp.symbol, int(round(sp.oxi_state)) if getattr(sp, "oxi_state", None) is not None else None) for sp in lst))
                    for lst in all_site_species]

        key_to_sp = {}
        for k, lst in zip(site_keys, all_site_species):
            key_to_sp.setdefault(k, lst)
        positional_disorder = False
        rads = {k: mean_site_radius(key_to_sp[k]) for k in key_to_sp}
        pair_thresh = { (a,b): 0.5*(rads[a]+rads[b]) for a,b in combinations_with_replacement(rads, 2) }
        r_thresh_max = max(1.0, max(pair_thresh.values(), default=0.0))
        neighbors = crystal.get_all_neighbors(r_thresh_max)
        intersect_orbits = []
        for i, nbrs in enumerate(neighbors):
            for nbr in nbrs:
                j = nbr.index
                if i >= j: 
                    continue
                d = getattr(nbr, "nn_distance", None) or nbr.distance
                ki, kj = site_keys[i], site_keys[j]
                pk = (ki, kj) if (ki, kj) in pair_thresh else (kj, ki)
                val = pair_thresh.get(pk, None)
                if val is None:
                    raise ValueError(
                        f"No threshold found for species pair {pk} (site indices {i},{j}) in {entry.CollectionCode}. "
                    )
                thresh = max(1.0, val)
                if d < thresh:
                    intersect_orbits.append((entry.CollectionCode, all_site_labels[i], tuple(sp.name for sp in key_to_sp[ki]), all_site_labels[j], tuple(sp.name for sp in key_to_sp[kj]), round(d,5), thresh))
        positional_disorder = len(intersect_orbits) > 0
        intersect_orbits = list(set(intersect_orbits)) 

        intersect_dicts = []  
        for t in intersect_orbits:
            intersect_dicts.append({
                "CollectionCode": t[0],
                "intersect_site1": t[1], 
                "intersect_site1_elements": t[2], 
                "intersect_site2": t[3], 
                "intersect_site2_elements": t[4], 
                "distance": t[5], 
                "threshold": t[6],
            })
        
        return positional_disorder, intersect_dicts

    except Exception as e:
        raise RuntimeError(f"CollectionCode {entry.CollectionCode}: intersect_orb failed: {e}") from e

def orb_keys_max_diff(key1, key2, prune_tol=1e-2, frac_tolerance = 1e-4):
    ndigits = max(0, -int(math.floor(math.log10(frac_tolerance))))
    if len(key1) != len(key2):
        return math.inf
    max_diff = 0.0
    for p, q in zip(key1, key2):
        d0 = abs(p[0] - q[0]); d0 = min(d0, 1.0 - d0)
        d1 = abs(p[1] - q[1]); d1 = min(d1, 1.0 - d1)
        d2 = abs(p[2] - q[2]); d2 = min(d2, 1.0 - d2)
        pt_max = max(d0, d1, d2)
        if pt_max > max_diff:
            max_diff = pt_max
        if max_diff > prune_tol:
            return np.round(max_diff, ndigits)
    return np.round(max_diff,ndigits)

def disorder_label(entry, *, site_tolerance: float = 0.0001, vac_tolerance: float = 1e-2, occ_tolerance: float = 1.0, frac_tolerance: float = 0.0001, verbose = False):
    """
    Accepts an ICSDEntry instance.
    Determining disorder types (vacancy, positional);
    Return list of disordered;
    Computes the degree of mixing (DOM) and fractional of disordered sites against total sites; 
    Converts the Wyckoff set into a canonical standard format; 
    Assembles a disorder_label as {wyckoff_set_std}_{space_group_number}_{element_seq}
    (e.g., 'j_i2_h_a_225_A_2B_{A+B}_C') for unique disorder structural identification.
    """
    orb_none = []
    wyck_float_warn = []
    occ_err = []
    is_occ_err = False
    equivalent_but_distinct = []
    merged_same_valence = []

    space_group_number = entry.spg_num
    code = entry.CollectionCode
    all_labels = entry.labels
    all_points = entry.points
    all_type_symbols = entry.type_symbols
    all_occ = entry.occ
    all_mults = entry.mults
    wyckoffs = entry.wyckoffs
    sym_ops = entry.sym_ops

    groups = defaultdict(list)
    disordered_type_list = []
    disordered_label_list = []
    sequence_map = {}
    total_sites = 0

    #====================== group index by Wyckoff letter ======================
    for idx, key in enumerate(wyckoffs):
        groups[key].append(idx)
    #====================== process each letter group ============================
    is_disorder = False
    is_sub_disorder = False
    is_vac_disorder = False
    for letter, idxs in groups.items():
        if verbose:
            print(f">>> Processing letter group '{letter}':")        
        points = [all_points[i] for i in idxs]  # for each letter group and their index, retrieve all the wyckoff positions belonging to this letter group
        occs = [all_occ[i] for i in idxs]
        equiv_orbit_groups = {}
        for local_idx, point in enumerate(points): #local_idx means the wyckoff orbits that share same wyckoff letter are given a local index
            mult_point = all_mults[idxs[local_idx]]
            label_point = all_labels[idxs[local_idx]]
            if occs[local_idx] <= 0:
                if verbose:
                    print(f"    → skipping orbit with local_index {local_idx}: {label_point} has zero/negative occupancy")
                continue            
            try:
                orb_key = get_equiv_positions(point, sym_ops, frac_tolerance = frac_tolerance) 
                if len(orb_key) > mult_point:
                    orb_key = get_equiv_positions(point, sym_ops, frac_tolerance*10)
                    wyck_float_warn.append({
                        "CollectionCode": code,
                        "label": label_point,
                        "wyckoff_position": point,
                        "equivalent_positions": len(orb_key),
                        "multiplicity": mult_point
                        })
                    if len(orb_key) > mult_point:
                        if verbose is True:
                            warnings.warn(
                                    f"Incorrect stoichiometry: \nCollectionCode {code}: labels= {label_point}, coordinate= {point}\n"
                                    f"equivalent positions {len(orb_key)} do not match with multiplicity {mult_point} dut to float error\n",
                                    stacklevel=2)

            except Exception:
                orb_none.append({
                    "CollectionCode": code,
                    "wyckoff_position": point
                    }) 
                continue
            for existing_key in list(equiv_orbit_groups.keys()):
                if orb_keys_max_diff(existing_key, orb_key, prune_tol=site_tolerance, frac_tolerance=frac_tolerance) <= site_tolerance:
                    equiv_orbit_groups[existing_key].append(local_idx)  # if max difference between two orb_key < site_tolerance, there are wyckoff orbits occupy same site, will be added into equiv_orbit_group
                    break
            else:
                equiv_orbit_groups[orb_key] = [local_idx]
        # Skip wyckoff letters that only contained zero/negative occupancy entries
        if not equiv_orbit_groups:
            if verbose:
                print("    → all orbits in this letter group have non-positive occupancy, skipping")
            continue
    #==================section 2: identifying type of site-disorder, calculating number of total/disordered sites, and degree of mixing for each disordered position===============            
        elem_counter = Counter()
        for local_idxs in equiv_orbit_groups.values():            
            labels = [all_labels[idxs[i]] for i in local_idxs]
            if verbose:
                print(f"    → checking orbit with local_index {local_idxs}: {labels}")
            elems = [''.join(filter(str.isalpha, lab)) for lab in labels]
            vals = [all_type_symbols[idxs[i]] for i in local_idxs]
            occs = [all_occ[idxs[i]] for i in local_idxs]
            mults = [all_mults[idxs[i]] for i in local_idxs]
            ndigits = max(0, -int(math.log10(frac_tolerance))) + 1
            coords = [points[i] for i in local_idxs]
            rounded_coords = [tuple(round(v, ndigits) for v in coord) for coord in coords]
            assert all(m == mults[0] for m in mults), "Inconsistent multiplicities in same orbit"
            multiplicity = mults[0]     #<<<-----------------------
            total_sites += multiplicity  #<<<-----------------------
            #---------detecting vacancy disorder-------------------
            occ_sum = sum(occs)  
            if occ_sum > occ_tolerance:
                is_occ_err = True
                occ_err.append({
                    "CollectionCode": code,
                    "label": sorted(labels),
                    "wyckoff_positions": rounded_coords,
                    "occupancies": occs,
                    "occ_sum": occ_sum
                    })
            vacancy_amount = 1.0 - occ_sum
            vacancy_amount = max(0.0, vacancy_amount) #all occ_sum > occ_tolerance will be rescaled down to 1.0 when calculating vacancy_amount
            if verbose:
                print(f"           vacancy_component = {vacancy_amount}")
                print(f"           total occupancy = {occ_sum}")
            has_vacancy = (vacancy_amount >= vac_tolerance)

            if len(local_idxs) > 1 or has_vacancy:
                if has_vacancy:
                    labels.append('VAC')
                    elems.append('VAC')
                    vals.append('VAC')
                    occs.append(1.0 - occ_sum)
                    is_vac_disorder = True
                    is_disorder = True
            #------------------------------------------------
                if verbose:
                    print(f"     !!! mixed elements, marking disorder for global idxs {[idxs[i] for i in local_idxs]}")

                #rounded_coords = [tuple(round(v / frac_tolerance) * frac_tolerance for v in coord) for coord in coords]
                if len(set(rounded_coords)) > 1:  #record warning 1: equivalent wyckoff orbit with different representative coord
                    equivalent_but_distinct.append({
                        "CollectionCode": code,
                        "label": sorted(labels),
                        "wyckoff_positions": rounded_coords,
                        "occupancies": occs,
                        "occ_sum": occ_sum
                        })
                    # warnings.warn(
                    #         f"WARNING: CollectionCode {CollectionCode} shows equivalent wyckoff orbit on one site but with different representative coordinate"
                    #         f"labels={labels}, coordinate={rounded_coords}",
                    #         stacklevel=2
                    #         )

                occ_dict = defaultdict(float)
                for el, occ in zip(vals, occs):
                    occ_dict[el] += occ
            #---------detecting substirutional disorder-------------------
                actual_species = [key for key in occ_dict if key != 'VAC']
                if len(actual_species) > 1:
                    is_sub_disorder = True
            #---------record outlier same valence species occupy site------ 
                has_dup_type_elems = len(vals) != len(set(vals)) 
                if has_dup_type_elems:   ##record warning 2: same elem with same valence on site-disorder
                    merged_same_valence.append({
                        "CollectionCode": code,
                        "labels": sorted(labels),
                        "valences": sorted(vals),
                        "wyckoff_positions": rounded_coords
                    })
                    # warnings.warn(
                    #         f"WARNING: CollectionCode {CollectionCode} has same elements with same valence state appears multiple times on one orbit"
                    #         f"labels={labels}, vals={vals}",
                    #         stacklevel=2
                    #         )

                if len(occ_dict) > 1:  # only different type_symbols will be accounted, this exclude same-element disorder,
                    is_disorder = True
                    if verbose:
                        print(f"           occ_dict: {occ_dict}")
                    disordered_label_list.append(sorted(labels))
                     #valence disorder will be implicit merge into one element
                    if len(set(elems)) == len(elems):
                        disordered_type_list.append(sorted(labels))
                    else:
                        disordered_type_list.append(sorted(vals))

    #====================section 3: get wyckoff sequence and elemental sequence =========================
            cnts = Counter(set(elems))  # dictionary counter {"O":3, "Li":1}, element as key, number as value, note, here we use set() to normalize {Fe2+: 1, Fe3+: 1} valence disorder into counter{Fe: 1}
            if len(cnts) == 1: # no site-disorder, single‑element orbit -> "3O" or "Li"
                elem, c = cnts.most_common(1)[0]
                orbit_str = f"{c}{elem}" if c > 1 else elem
            else: # multi‑element orbit (site‑disorder) -> sorted then "Li+Mn"
                parts = []
                for el in sorted(cnts): 
                    parts.append(f"{cnts[el]}{el}" if cnts[el] > 1 else el)
                orbit_str = "+".join(parts)
            elem_counter[orbit_str] += 1
        
        elem_parts = [] #elem_part is the elemental correspondings of each wyckoff letter group
        for orbit_str, cnt in elem_counter.items():
            if cnt == 1:
                elem_parts.append(orbit_str)
            else:
                if "+" in orbit_str:
                    elem_parts.append(f"{cnt}({orbit_str})")
                else:
                    elem_parts.append(f"{cnt}{orbit_str}")
        elem_parts = sorted(elem_parts)
        if verbose:
            print(f"      elem_parts: {elem_parts}")

        # constructing wyckoff_sequence
        n_orbits = len(equiv_orbit_groups)
        wyckoff_part = letter + (str(n_orbits) if n_orbits > 1 else "")
        # single orbit with single element has no {}, otherwise with {}
        single_orbit_single_elem = (
            n_orbits == 1
            and len(elem_parts) == 1
            and "+" not in elem_parts[0]
            and not elem_parts[0][0].isdigit()
        )
        if single_orbit_single_elem:
            elem_part = elem_parts[0]
        else:
            elem_part = "{" + ",".join(elem_parts) + "}"

        sequence_map[wyckoff_part] = elem_part 
    #sequence_map = dict(sorted(sequence_map.items(), key=lambda kv: kv[0]))
    #sequence_map = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
    #wyckoff_seq = "_".join(sequence_map.keys())
    #element_seq = "_".join(sequence_map.values())
    #=========================section 3.2: calculating fraction_of_disordered_sites, and degree_of_mixing==================================
    if disordered_label_list:
        fraction_of_disordered_sites, degree_of_mixing =  compute_disorder(entry, disordered_label_list, total_sites)
    else:
        fraction_of_disordered_sites = 0
        degree_of_mixing = None
    # DOM is only comparable when applying to similar structures (e.g. same prototype, same disordered positions, with different mixing ratio!!!)
    #=========================section 4: get canonical wyckoff sequence=======================  
    #wyckoff_set_std = get_canonical_wyckoff_sets(sequence_map, mapping_by_sg, entry.spg_num)
    #disorder_label = f"{wyckoff_set_std}_{space_group_number}_{element_seq}"      
    wyckoff_set_std, element_seq, sequence_map_std = _canonicalize_sequence_map_with_hall(
        sequence_map, entry.spg_num
    )
    disorder_label = f"{wyckoff_set_std}_{space_group_number}_{element_seq}"

    is_positional_disorder = False
    intersect_orbs = []
    intersect_orb_error= None
    if not is_occ_err:
        try:
            is_positional_disorder, intersect_orbs = intersect_orb(entry, site_tolerance= site_tolerance, occ_tolerance= occ_tolerance)
        except Exception as e:
            is_positional_disorder = None
            intersect_orb_error = str(e)
    else:
        is_positional_disorder = None

    return {
        #disordered type:
        "is_disorder": is_disorder,
        "is_vac_disorder": is_vac_disorder,
        "is_sub_disorder": is_sub_disorder,
        "is_positional_disorder": is_positional_disorder,

        #disordered sites list:
        "disordered_valence_list": disordered_type_list,
        "disordered_label_list": disordered_label_list,

        #computed disordered info:
        "fraction_of_disordered_sites": fraction_of_disordered_sites,
        "degree_of_mixing": degree_of_mixing,

        #label info:
        "sequence_map": sequence_map_std,
        "wyckoff_set_std": wyckoff_set_std,
        "disorder_label": disorder_label,

        #outliers:
        "wyck_float_warn": wyck_float_warn,
        "occ_err_sites": occ_err,
        "equivalent_but_distinct_sites": equivalent_but_distinct,
        "same_valence_sites": merged_same_valence,
        "intersect_orbs": intersect_orbs,
        "orb_none_entry": orb_none,
        "intersect_orb_error": intersect_orb_error
    } 


def get_sword_label(
    data: Union[str, Structure],  # CIF text / CIF path / pymatgen Structure
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    occupancy_tolerance: float = 1.0,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    conventional_struct: bool = True,
    refine_struct: bool = False,
) -> str:
    if isinstance(data, Structure):
        cif_txt = data.to(fmt="cif")
    elif isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                cif_txt = f.read()
        else:
            cif_txt = data
    else:
        raise TypeError("data must be str (CIF text/path) or pymatgen.core.Structure")

    entry = StructureEntry.from_txt(
        cif_txt,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        occupancy_tolerance=occupancy_tolerance,
        conventional_struct=conventional_struct,
        refine_struct=refine_struct,
    )

    return disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        verbose=False,
    )["disorder_label"]

def get_sword_info(
    data: Union[str, Structure],  # CIF text / CIF path / pymatgen Structure
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    occupancy_tolerance: float = 1.0,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    conventional_struct: bool = True,
    refine_struct: bool = False,
) -> str:
    if isinstance(data, Structure):
        cif_txt = data.to(fmt="cif")
    elif isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                cif_txt = f.read()
        else:
            cif_txt = data
    else:
        raise TypeError("data must be str (CIF text/path) or pymatgen.core.Structure")

    entry = StructureEntry.from_txt(
        cif_txt,
        source="SWORD",
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        occupancy_tolerance=occupancy_tolerance,
        conventional_struct=conventional_struct,
        refine_struct=refine_struct,
    )

    return entry, disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        verbose=False,
    )

def find_parent_ICSD(child, icsd_df, symprec_child=1e-1, symprec_search=1.0): #symprec_search should be high enough to recover the higher symmetry
    child_label = get_sword_label(child, symprec=symprec_child, conventional_struct=True)
    elements = sorted([el.symbol for el in child.composition.elements])
    mask_sets = [
        comb
        for k in range(2, len(elements) + 1)
        for comb in combinations(elements, k)
    ]

    out = []
    for mask in mask_sets:
        masked = child.copy()
        masked.replace_species({el: DummySpecie("X", 0) for el in mask})
        label = get_sword_label(masked, symprec=symprec_search, conventional_struct=True)

        mix = "+".join(sorted(mask))
        label = label.replace("X", f"{{{mix}}}")

        rows = icsd_df[icsd_df["disorder_label"] == label]
        out.append({
            "child_label": child_label,
            "parent_label": label,
            "matched_labels": rows["disorder_label"].tolist(),
            "id": rows["CollectionCode"].tolist(),
        })
    return out

def get_sword_label_for_ICSD(
    collection_code,
    *,
    ICSD_df,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    meta=None,
) -> str:
    entry = StructureEntry.from_collection_code(
        collection_code,
        ICSD_df=ICSD_df,
        meta=meta,
    )
    return disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        verbose=False,
    )["disorder_label"]

def get_sword_info_for_ICSD(
    collection_code,
    *,
    ICSD_df,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    meta=None,
) -> str:
    entry = StructureEntry.from_collection_code(
        collection_code,
        ICSD_df=ICSD_df,
        meta=meta,
    )
    return entry, disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        verbose=False,
    )

def get_sword_label_from_pyxtal(pyxtal_dict):
    """
    Generates a SWORD label from a PyXtal dictionary representation.

    Args:
        pyxtal_dict (dict): A dictionary containing 'group' (int), 
                            'sites' (list of lists of strings), 
                            and 'species' (list of strings).

    Returns:
        str: The standardized SWORD label.
    """
    sg_num = int(pyxtal_dict['group'])
    species_list = pyxtal_dict['species']
    sites_list = pyxtal_dict['sites']
    
    # 1. Group elements by Wyckoff letter
    # Flatten the PyXtal structure: Wyckoff Letter -> List of elements on that site
    wyck_groups = defaultdict(list)
    
    for species, site_sublist in zip(species_list, sites_list):
        for site_str in site_sublist:
            # Extract just the letter (e.g., '4h' -> 'h', '18e' -> 'e')
            letter = "".join(filter(str.isalpha, site_str))
            wyck_groups[letter].append(species)
            
    # 2. Construct the sequence map (Key: WyckoffPart, Value: ElementPart)
    sequence_map = {}
    
    for letter, elems in wyck_groups.items():
        count = len(elems) # Number of distinct orbits (sites) for this letter
        
        # Build Wyckoff Part (e.g., 'a' or 'a12')
        wyck_key = f"{letter}{count}" if count > 1 else letter
        
        # Build Element Part
        # Count element occurrences: e.g. {'Sn': 5, 'Te': 6, 'Ag': 1}
        cnts = Counter(elems)
        parts = []
        for el, n in cnts.items():
            if n > 1:
                parts.append(f"{n}{el}")
            else:
                parts.append(el)
        
        # Sort parts as strings (standard lexicographical sort matches SWORD format: 10 < 2, 5 < 6 < A)
        parts.sort()
        
        # Determine formatting based on orbit count
        # If multiple sites exist for this letter, enclose in brackets {}
        if count == 1:
            elem_val = parts[0]
        else:
            elem_val = "{" + ",".join(parts) + "}"
            
        sequence_map[wyck_key] = elem_val

    # PyXtal-generated sequences have some canonicalization issues; bypassing for now (need to investigate further)
    wyckoff_set_std, element_seq, _ = _canonicalize_sequence_map_with_hall(
        sequence_map, sg_num, return_trace=False
    )
    return f"{wyckoff_set_std}_{sg_num}_{element_seq}"
