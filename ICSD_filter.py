import json
import math
import pandas as pd
from io import StringIO
import re
from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Structure
from tqdm import tqdm
import warnings
import numpy as np
import itertools
from itertools import combinations_with_replacement
from collections import Counter, defaultdict

ICSD = pd.read_pickle('/home/users/yyhuang/ICSD/deduplicate/ICSD2024_summary_w_spg_info.pkl')
ICSD_valid = pd.read_pickle('/home/users/yyhuang/ICSD/deduplicate/job_script/ICSD_filter/ICSD_screened.pkl')
wyckoff_sets = pd.read_json('/home/users/yyhuang/ICSD/deduplicate/wyckoff_sets.json')
radius_df = pd.read_csv('/home/users/yyhuang/ICSD/deduplicate/all_radii.csv')

def clean_num(token: str) -> float:
    base = re.sub(r"\(.*\)", "", token)
    try:
        return float(base)
    except Exception:
        return None

class ICSDEntry:
    """
    The ICSD CIF wrapper:
    The instance can be constructed from_cif, or from_collection_code in ICSD_df(DataFrame)
    meta: optional dict (e.g. {'CollectionCode':5004,'StructuredFormula':'Na2O'})
    """
    def __init__(self, cif_text,collection_code = None, meta=None):
        self.meta = meta
        if collection_code is not None:
            self.CollectionCode = collection_code
        else:
            self.CollectionCode = None

        cif_str = StringIO(cif_text)
        parser = CifParser(cif_str)
        cif_dict = parser.as_dict()
        block_name = list(cif_dict.keys())[0]
        block = cif_dict[block_name]
        try:
            space_group_number = int(re.sub(r"\(.*\)", "", block["_space_group_IT_number"]))
        except Exception:
            space_group_number = block.get("_space_group_IT_number", None)

        a = block["_cell_length_a"]
        b = block["_cell_length_b"]
        c = block["_cell_length_c"]

        xs = [clean_num(x) for x in block["_atom_site_fract_x"]]
        ys = [clean_num(x) for x in block["_atom_site_fract_y"]]
        zs = [clean_num(x) for x in block["_atom_site_fract_z"]]
        if any(v is None for v in xs + ys + zs):
            warnings.warn(f"CollectionCode {self.CollectionCode}: coordinate contains non-standard value(s).")

        labels = block["_atom_site_label"]
        type_symbols = block["_atom_site_type_symbol"]
        occupancies = [clean_num(x) for x in block["_atom_site_occupancy"]]
        if any(v is None for v in occupancies):
            warnings.warn(f"CollectionCode {self.CollectionCode}: site_occupancy contains non-standard value(s).")

        mults = [int(m) for m in block["_atom_site_symmetry_multiplicity"]]
        wycks = block["_atom_site_Wyckoff_symbol"]

        sym_ops = block.get("_space_group_symop_operation_xyz", [])
        sym_ops = [SymmOp.from_xyz_str(op_str) for op_str in sym_ops]

        nsites = len(labels)
        points = []
        for i in range(nsites):
            coord = (xs[i], ys[i], zs[i])
            points.append(coord)

        records = []
        for i, lab in enumerate(labels):
            coord = [xs[i], ys[i], zs[i]]
            rec = {
                'label'         : lab,
                'type_symbol'   : type_symbols[i],
                'multiplicity'  : int(mults[i]),
                'wyckoff_symbol': wycks[i],
                'coordinate'    : coord,
                'occupancy'     : occupancies[i], 
            }
            records.append(rec)
            
        self.cif_dict = cif_dict
        self.cif_str = cif_text
        self.spg_num = space_group_number
        self.lattice = (a,b,c)
        self.xs = xs; self.ys = ys; self.zs = zs
        self.labels = labels
        self.type_symbols = type_symbols
        self.occ = occupancies 
        self.mults = mults
        self.wyckoffs = wycks
        self.sym_ops = sym_ops
        self.points = points
        self.records = records

    @classmethod
    def from_cif(cls, cif_text: str = None, meta=None):
        return cls(cif_text, meta=meta, collection_code = None)
    
    @classmethod
    def from_collection_code(cls, collection_code, ICSD_df: pd.DataFrame = None, meta=None):
        row = ICSD_df[ICSD_df['CollectionCode'] == collection_code].iloc[0]
        cif_text = row['cif']
        if meta == 'all':
            meta = row.to_dict()
        return cls(cif_text, meta=meta, collection_code = collection_code)
    
def periodic_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff_pbc = diff - np.round(diff)
    return np.linalg.norm(diff_pbc)

def get_equiv_positions(point, sym_ops, frac_tolerance=1e-4):
    p0 = np.asarray(point, dtype=float) % 1.0
    orb_raw = []
    for op in sym_ops:
        if hasattr(op, "operate"):
            p = op.operate(p0)
        else:
            p = op(p0)
        orb_raw.append(p % 1.0) 

    clusters = [] 
    for p_new in orb_raw:
        found = False
        for rep, points_in_cluster in clusters:
            if periodic_dist(p_new, rep) <= frac_tolerance*2:
                points_in_cluster.append(p_new)
                found = True
                break

        if not found:
            clusters.append([p_new.copy(), [p_new.copy()]])

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

def compute_disorder(entry, disordered_list, total_sites = 0, verbose = False):
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

def get_canonical_wyckoff_sets(sequence_map, mapping_by_sg, space_group_number):
    wyckoff_set_std = ''
    equivalent_sequence_list = []

    if isinstance(sequence_map, dict) and sequence_map:
        seq_expanded = [
            letter
            for key in sequence_map.keys()
            for letter, num in re.findall(r'([A-Za-z]+)(\d*)', key)
            for _ in range(int(num) if num else 1)
        ]

        wp_mappings = mapping_by_sg.get(space_group_number, {})

        if wp_mappings:
            for wp in wp_mappings.values():
                letter_map = {}
                for idx, ch in wp.items():
                    letter_map.setdefault(ch, []).append(idx)
                
                try:
                    indices = [letter_map[ch] for ch in seq_expanded]
                except KeyError:
                    continue
                equivalent_sequence_list.extend(itertools.product(*indices))

            if equivalent_sequence_list:
                best_tuple = min(equivalent_sequence_list, key=sum)
                first_wp_no = min(wp_mappings)
                idx_to_letter = wp_mappings[first_wp_no]
                letter_seq = [idx_to_letter.get(idx, '?') for idx in best_tuple]
                letter_counts = Counter(letter_seq)
                used = Counter()
                result = []
                
                for ch in letter_seq:
                    used[ch] += 1
                    count = letter_counts[ch]
                    
                    if count > 1:
                        if used[ch] == 1:
                            result.append(f"{ch}{count}")
                    else:
                        result.append(ch)
                        
                wyckoff_set_std = '_'.join(result)
    
    return wyckoff_set_std

def mean_site_radius(species_list):
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
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crystal = Structure.from_str(entry.cif_str, fmt="cif",
                                        site_tolerance = site_tolerance,
                                        occupancy_tolerance = occ_tolerance
                                        )
        all_site_labels = [s.label for s in crystal]
        all_site_species = [list(s.species.keys()) for s in crystal]
        site_keys = [tuple(sorted((sp.element.symbol, int(round(sp.oxi_state)) if getattr(sp, "oxi_state", None) is not None else None) for sp in lst)) for lst in all_site_species]
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
    

def disorder_label(entry, site_tolerance: float = 0.0001, vac_tolerance: float = 1e-1, occ_tolerance: float = 1.0, frac_tolerance: float = 0.0001, verbose = False):
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
            if occs[local_idx] < 0:
                continue            
            try:
                orb_key = get_equiv_positions(point, sym_ops, frac_tolerance = frac_tolerance) 
                if len(orb_key) > mult_point:
                    wyck_float_warn.append({
                        "CollectionCode": code,
                        "site": label_point,
                        "wyckoff_position": point,
                        "equivalent_positions": len(orb_key),
                        "multiplicity": mult_point
                        })
                    orb_key = get_equiv_positions(point, sym_ops, frac_tolerance*10)
                    if len(orb_key) > mult_point:
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
                    "site": sorted(labels),
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

                if len(set(rounded_coords)) > 1:  #record warning 1: equivalent wyckoff orbit with different representative coord
                    equivalent_but_distinct.append({
                        "CollectionCode": code,
                        "site": sorted(labels),
                        "wyckoff_positions": rounded_coords,
                        "occupancies": occs,
                        "occ_sum": occ_sum
                        })

                occ_dict = defaultdict(float)
                for el, occ in zip(vals, occs): #valence disorder will be implicit merge into one element
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
                        "site": sorted(labels),
                        "valences": sorted(vals),
                        "wyckoff_positions": rounded_coords
                    })

                if len(occ_dict) > 1:  # only different type_symbols will be accounted, this exclude same-element disorder,
                    is_disorder = True
                    if verbose:
                        print(f"           occ_dict: {occ_dict}")
                    disordered_label_list.append(sorted(labels))

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
    sequence_map = dict(sorted(sequence_map.items(), key=lambda kv: kv[0]))
    element_seq = "_".join(sequence_map.values())
    #=========================section 3.2: calculating fraction_of_disordered_sites, and degree_of_mixing==================================
    if disordered_label_list:
        fraction_of_disordered_sites, degree_of_mixing =  compute_disorder(entry, disordered_label_list, total_sites)
    else:
        fraction_of_disordered_sites = 0
        degree_of_mixing = None

    #=========================section 4: get canonical wyckoff sequence=======================  
    wyckoff_set_std = get_canonical_wyckoff_sets(sequence_map, mapping_by_sg, entry.spg_num)
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
        "sequence_map": sequence_map,
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

#codes_to_select = [33404, 38870, 83283, 148749]
#sample = ICSD[ICSD['CollectionCode'].isin(codes_to_select)]
#database = sample.copy()
database = ICSD_valid.copy()
new_cols = [
    "is_disorder",
    "is_sub_disorder",
    "is_vac_disorder",
    "is_positional_disorder",
    "fraction_of_disordered_sites",
    "degree_of_mixing",
    "disorder_label",
    "wyckoff_set_std",
    "disordered_valence_list",
]

out_lists = {c: [] for c in new_cols}
wyck_float_warn = []
main_errors = []
intersect_orb_errors = []

occ_err_sites = []
equivalent_but_distinct_sites = []
same_valence_sites = []
intersect_orb_sites = []
orb_none_sites = []

for _, row in tqdm(database.iterrows(), total=len(database), desc="prescreening_ICSD"):
    entry = ICSDEntry.from_collection_code(row["CollectionCode"], database)
    try:
        results = disorder_label(entry, site_tolerance= 1e-4, occ_tolerance= 1.05, vac_tolerance = 1e-2, frac_tolerance = 1e-4)
        for c in new_cols:
            out_lists[c].append(results.get(c, None))

        lst = results.get("wyck_float_warn")
        if lst:
            wyck_float_warn.extend(results.get("wyck_float_warn", []))

        lst = results.get("occ_err_sites")
        if lst:
            occ_err_sites.extend(results.get("occ_err_sites", []))
            
        lst = results.get("equivalent_but_distinct_sites")
        if lst:
            equivalent_but_distinct_sites.extend(results.get("equivalent_but_distinct_sites", []))

        lst = results.get("same_valence_sites")    
        if lst:
            same_valence_sites.extend(results.get("same_valence_sites", []))

        lst = results.get("intersect_orbs")
        if lst:
            intersect_orb_sites.extend(results.get("intersect_orbs", []))

        lst = results.get("intersect_orb_error") 
        if lst:
            intersect_orb_errors.append(results.get("intersect_orb_error", []))
            
        lst = results.get("orb_none_entry") 
        if lst:
            orb_none_sites.append(results.get("orb_none_entry", []))

    except Exception as e:
        out_lists[c].append(None)
        main_errors.append((entry.CollectionCode, str(e)))

n_rows = len(database)
for c in new_cols:
    ln = len(out_lists[c])
    if ln != n_rows:
        print(f"COL {c}: length {ln} (diff {n_rows - ln})")
    else:
        database[c] = out_lists[c]

import os
BASE_DIR = '/home/users/yyhuang/ICSD/deduplicate/job_script/ICSD_filter/formal_run/'

path = os.path.join(BASE_DIR, 'ICSD_valid_dedup_Stol_1e-4_vtol_1e-2_occ_tol_1.05.pkl')
database.to_pickle(path)

if orb_none_sites:
    path = os.path.join(BASE_DIR, 'orb_none_sites.json')
    with open(path, 'w') as f:
        json.dump(orb_none_sites, f, ensure_ascii=False, indent=2)
    print(f"{len(orb_none_sites)} Sites skipped due to 'orb is None'")

if wyck_float_warn:
    wyck_float_warn = pd.DataFrame(wyck_float_warn)
    print(f"{len(wyck_float_warn['CollectionCode'].unique())} Entries has floating-point error in wyckoff positions")
    path = os.path.join(BASE_DIR, 'wyck_float_warn.csv')
    wyck_float_warn.to_csv(path, index = False)

if occ_err_sites:
    occ_err_sites = pd.DataFrame(occ_err_sites)
    path = os.path.join(BASE_DIR, 'occ_err_sites.csv')
    occ_err_sites.to_csv(path, index = False)
    print(f"{len(occ_err_sites['CollectionCode'].unique())} Entries has occ > occ_tolerance")

if equivalent_but_distinct_sites:
    equivalent_but_distinct_sites = pd.DataFrame(equivalent_but_distinct_sites)
    path = os.path.join(BASE_DIR, 'equivalent_but_distinct_sites.csv')
    equivalent_but_distinct_sites.to_csv(path, index = False)
    print(f"{len(equivalent_but_distinct_sites['CollectionCode'].unique())} Entries contain site-disorder with equivalent but distinct coordiate")

if same_valence_sites:
    same_valence_sites = pd.DataFrame(same_valence_sites)
    path = os.path.join(BASE_DIR, 'same_valence_sites.csv')
    same_valence_sites.to_csv(path, index = False)
    print(f"{len(same_valence_sites['CollectionCode'].unique())} Entries contain same elems on one orbit and is merged into one elems")

if intersect_orb_sites:
    intersect_orb_sites = pd.DataFrame(intersect_orb_sites)
    path = os.path.join(BASE_DIR, 'intersect_orb_sites.csv')
    intersect_orb_sites.to_csv(path, index = False)
    print(f"{len(intersect_orb_sites['CollectionCode'].unique())} Entries contain positional disorder")

if main_errors:
    path = os.path.join(BASE_DIR, 'main_errors.json')
    with open(path, 'w') as f:
        json.dump(main_errors, f, ensure_ascii=False, indent=2)
    print(f"The following {len(main_errors)} entries has error:")
    for code, msg in main_errors:
        print(f" ColletionCode{code}: {msg}")

if intersect_orb_errors:
    path = os.path.join(BASE_DIR, 'intersect_orb_errors.json')
    with open(path, 'w') as f:
        json.dump(intersect_orb_errors, f, ensure_ascii=False, indent=2)
    print(f"The following {len(intersect_orb_errors)} Entries have error when processing intersect orbit:")
    for msg in intersect_orb_errors:
        print(msg)
else:
    print("completed")
