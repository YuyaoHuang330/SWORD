import re

import pandas as pd
from importlib.resources import files

_DATA_DIR = files("sword") / "data"
_WYCKOFF_SETS_JSON = _DATA_DIR / "wyckoff_sets.json"
_HALL_MAPS_JSON = _DATA_DIR / "hall_letter_maps_by_sg.json"

wyckoff_sets = pd.read_json(_WYCKOFF_SETS_JSON)


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

#_SWORD_DIR = os.path.dirname(__file__)
#_HALL_WP_JSON = os.path.join(_SWORD_DIR, "sword_hall_wp_perms_by_sg.json")
#_BUILD_SCRIPT = os.path.join(_SWORD_DIR, "build_sword_hall_wp_table.py")
#_HALL_MAPS_JSON = os.path.join(_SWORD_DIR, "hall_letter_maps_by_sg.json")

#_SWORD_HALL_WP_PERMS = None

# def _build_hall_wp_table_if_missing():
#     if os.path.exists(_HALL_WP_JSON):
#         return

#     cmd = [
#         sys.executable, _BUILD_SCRIPT,
#         "--wyckoff-sets", _WYCKOFF_SETS_JSON,
#         "--out", _HALL_WP_JSON,
#     ]
#     if os.path.exists(_HALL_MAPS_JSON):
#         cmd += ["--hall-maps", _HALL_MAPS_JSON]

#     subprocess.run(cmd, check=True)

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
