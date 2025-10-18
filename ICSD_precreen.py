import pandas as pd
from io import StringIO
import re
from pymatgen.io.cif import CifParser
from pymatgen.core.operations import SymmOp
from tqdm import tqdm
import warnings

ICSD = pd.read_pickle('/home/users/yyhuang/ICSD/deduplicate/ICSD2024_summary_w_spg_info.pkl')

def contains_H(entry) -> bool:
    """
    check if the ICSDentry contain any hydrogen or its isotopes D/T
    """
    if hasattr(entry, "labels"):
        labels = entry.labels
    else:
        return False
    elems_raw = [''.join(filter(str.isalpha, lab)) for lab in labels if lab]
    elems_set = set(elems_raw)

    return bool(elems_set & {'H', 'D', 'T'})


from pymatgen.core.periodic_table import Element
def get_non_elements(entry, excluded_elements=None) -> bool:
    """
    Remove symbol not in periodic table: {'M', 'D', 'T', 'X', 'L'}, 
    excluded_elements: remove user-defined unwanted element (excluded_elements) in periodic table
    """
    periodic_table = {e.symbol for e in Element}
    non_elem = []
    if not hasattr(entry, "labels"):
        return False, []
    
    if excluded_elements:
        excluded_elements = set(excluded_elements or [])
        known_elements = periodic_table - excluded_elements
    else:
        known_elements = periodic_table
    
    labels = entry.labels
    elems_raw = [''.join(filter(str.isalpha, lab)) for lab in labels if lab]
    elems_set = set(elems_raw)
    non_elem = [elem for elem in elems_set if elem not in known_elements]
    contains_non_element = bool(non_elem)
    return contains_non_element, non_elem


def check_cif_format(entry, occ_tolerance: float = 1.0):
    """
    check any non-standard format for each ICSD entry, such as:
    coordinate:    entry.xs = 0.485.078,
    wyckoff letter:     [a,b,f,e,*,*,g],
    occupancy: occupancy value > occ_tolerance or occupancy < 0, can only check occupancy of each wyckoff position, cannot sum up occupancy of disordered sites
    """
    coord_err = wyck_err = mult_err = occ_err = False
    if any(v is None for v in entry.xs) or any(v is None for v in entry.ys) or any(v is None for v in entry.zs):
        coord_err = True
    # --- Wyckoff ---
    if not hasattr(entry, "wyckoffs") or entry.wyckoffs is None:
        wyck_err = True
    else:
        wycks = [str(w) for w in entry.wyckoffs if w]
        if any(not w.isalpha() for w in wycks):
            wyck_err = True
    # --- multiplicity ---
    if not hasattr(entry, "mults") or entry.mults is None:
        mult_err = True
    else:
        mults = [str(m) for m in entry.mults if m]
        if any(not m.isdigit() for m in mults):
            mult_err = True
    # --- occupancy ---
    if not hasattr(entry, "occ") or entry.occ is None or any(o is None for o in entry.occ):
        occ_err = True
    else:
        try:
            err_occs_list = []
            occs = [float(o) for o in entry.occ]
            err_occs_list = [o for o in occs 
                             if o > occ_tolerance
                             or o < 0] #record specific occ value, can only check single wyckoff orbit
            occ_err = bool(err_occs_list)
        except Exception:
            occ_err = True

    return coord_err, wyck_err, mult_err, occ_err, err_occs_list


def clean_num(token: str) -> float:
    base = re.sub(r"\(.*\)", "", token)
    try:
        return float(base)
    except Exception:
        return None
    

class ICSDEntry:
    """
    The ICSD CIF wrapper: ICSDEntry use CifParser to read original CIF file and retrieve key crystallographic information via entry.attribute notation
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


coord_err_codes = []
wyck_err_codes = []
mult_err_codes = []
occ_err_codes = []
occ_err_list = []
non_elem_codes = []
non_elem_list = []
H_codes = []

for _, row in tqdm(ICSD.iterrows(), total=len(ICSD), desc="prescreening_ICSD"):
    entry = ICSDEntry.from_collection_code(row["CollectionCode"], ICSD)

    coord_err, wyck_err, mult_err, occ_err, occ_err_values = check_cif_format(entry, occ_tolerance= 1.05)
    if coord_err: coord_err_codes.append(entry.CollectionCode)
    if wyck_err:  wyck_err_codes.append(entry.CollectionCode)
    if mult_err:  mult_err_codes.append(entry.CollectionCode)
    if occ_err:   
        occ_err_codes.append(entry.CollectionCode)
        occ_err_list.append((entry.CollectionCode, occ_err_values))

    contain_non_elems, non_elems = get_non_elements(entry, ['He','Ne','Ar','Kr','Es'])
    if contain_non_elems:
        non_elem_codes.append(entry.CollectionCode)
        non_elem_list.append((entry.CollectionCode, non_elems))

    has_H = contains_H(entry)
    if has_H:
        H_codes.append(entry.CollectionCode)
        
    del entry

print(f"H/D/T-containing Entries: {len(H_codes)}")
print(f"Non-element containing Entries: {len(non_elem_codes)}")
print(f"Occupancy > occ_tolerance: {len(occ_err_codes)}")
print(f"Coordinate error codes: {len(coord_err_codes)}")
print(f"General Wyckoff assignment error: {len(wyck_err_codes)}")
print(f"Multiplicity error codes: {len(mult_err_codes)}")

unwanted_codes = set().union(*[H_codes, non_elem_codes, occ_err_codes, coord_err_codes, wyck_err_codes, mult_err_codes])
ICSD_screened = ICSD[~ICSD['CollectionCode'].isin(unwanted_codes)]

#BASE_DIR = #your output path
# path = os.path.join(BASE_DIR, 'name_of_icsd.pkl')
# database.to_pickle(path)