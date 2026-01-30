# SWORD——Symmetry and Wyckoff-sequence of Ordered and Disordered crystals

# How to Use SWORD:
`from utils_SWORD import get_sword_label`
`from utils_SWORD import get_sword_info`
1. return SWORD_label: 
- `label = get_sword_label(structure, symprec={your_preferred_symprec})`  #pymatgen.core.Structure
#or
- `label = get_sword_label(cif_text, symprec={your_preferred_symprec})`
#or
- `label = get_sword_label("path/to/file.cif", symprec={your_preferred_symprec})`

3. return all information
- `entry, dict = get_sword_info(structure, symprec={your_preferred_symprec})`  #pymatgen.core.Structure
#or
- `entry, dict = get_sword_info(cif_text, symprec={your_preferred_symprec})`
#or
- `entry, dict = get_sword_info("path/to/file.cif", symprec={your_preferred_symprec})`
- `display(entry.df)`

## ICSD-specific interfaces (for ICSD cleaning)
**Import**
`from utils_SWORD import get_sword_label_for_ICSD`  
`from utils_SWORD import get_sword_info_for_ICSD`

### 1) Return SWORD label only
`label = get_sword_label_for_ICSD(collection_code, ICSD_df=ICSD_df)`

### 2) Return full information
`entry, info = get_sword_info_for_ICSD(collection_code, ICSD_df=ICSD_df)`  
`display(entry.df)`

> Note: `collection_code` is the ICSD CollectionCode; `ICSD_df` is a DataFrame containing `CollectionCode` and `cif` columns.

## find_parent_ICSD
**Purpose**  
Finds the parent disordered ICSD entry from a given random ordered child structure and returns the parent candidate(s) SWORD label based on matching rules.

**Input**  
- `child`: ordered child structure (**same accepted types as `get_sword_label`** )  
- `ICSD_df`: ICSD database/DataFrame **with SWORD labels** (must include `CollectionCode`, and `disorder_label` columns)  
- other optional parameters, symprec_child: symprec of labelling ordered structures, symprec_search: symprec of searching & recovering parent label)

**Example**
`parent_code = find_parent_ICSD(child, ICSD_df=ICSD_df)`


# Introduction of Main functions:

**Classifying, labelling and filtering ICSD, identify high-quality and trustworthy ICSD entries**

<div id="note" style="border:1px solid #d0d7de; padding:16px; background:#fcffd2; border-radius:6px;">
<strong style="font-size:1.05em;">Known issues of ICSD</strong>

<p style="margin-top:0.5em;">
(1). Non-element symbols (e.g., {'M', 'D', 'T', 'X', 'L'});

(2). Occupancy larger than 1.0;

(3). Coordinates error (e.g. 0.2840.29018(16) - See ICSD CollectionCode 70589);

(4). Wyckoff letter error (e.g., ‘*’)
</p>

See details in [ICSD Pre-screening Report](./ICSD_prescreen_report.txt)
</div>

# 1. StructureEntry instance:

`StructureEntry`:

The CIF wrapper: `StructureEntry` use `Pymatgen CifParser` to read original CIF file and retrieve key crystallographic information via `entry.attribute` notation
The instance can be constructed by:

**From ICSD CollectionCode:**

`StructureEntry.from_CollectionCode(CollectionCode, ICSD: pd.Dtaframe = None)`
A valid `CollectionCode` is required, and the target entry must exist within the `ICSD` DataFrame you are using:

`example = StructureEntry.from_collection_code(137626, ICSD)`

**From CIF text:**

`example = StructureEntry.from_cif(cif_str)`

```
        **Frequently used entry.attribute:**
        self.cif_str                      #original CIF text
        self.CollectionCode               #CollectionCode
        self.spg_num                      #space group number
        self.lattic                       #lattice parameter (a,b,c)   
        self.points                       #list of all (x,y,z) sites' wyckoff positions in CIF
        self.xs = xs                      #list of all sites' x-axis coordinates in CIF
        self.ys = ys                      #list of all sites' y-axis coordinates in CIF
        self.zs = zs                      #list of all sites' z-axis coordinates in CIF
        self.labels                       #list of all sites' labels
        self.type_symbols                 #list of all sites' type_symbols
        self.occ                          #list of all sites' occupancies
        self.mults                        #list of all sites' multiplicity 
        self.wyckoffs                     #list of all sites' wyckoff letters
        self.sym_ops                      #list of all symmetry operations recorder in CIF
    
```

# 2. Pre-screening of ICSD:

In ICSD_prescreen.py or ICSD_prescree_filter.ipynb file :

Run the file (it will take about 7 mins):

**It will pre-screen ICSD based on following 3 functions:**

- `contains_H(entry)`:

  Remove Entries contain hydrogen and its isotopes D and T labels in CIF (note: The entry will not be removed if the atomic `labels` in the CIF do not include hydrogen, even if the element is claimed in either the ICSD `StructureFormula` or the CIF's `_chemical_formula_structural` tag.)

- `get_non_elements(entry, excluded_elements=None)`:

  Identify and remove unwanted elements (non-element symbols: 'M', 'X', 'L', or **any undesired element that you want to exclude**)

  In ICSD_prescreen.py file, we exclude the following noble gases and radioactive elements:

  ['He', 'Ne', 'Ar', 'Kr', 'Es'].

- `check_cif_format(entry, occ_tolerance: float = 1.0)` :

  Identify and remove entries contain any non-standard format, including:

  non-standard coordinate: 0.485.078
  non-standard wyckoff letters, such as ‘*’.
  occupancy: occupancy value > occ_tolerance (default as 1.0), or occupancy < 0. (note: can only identify the occupancy of individual site; it is unable to sum the occupancy values for disordered sites.)


# 3. Labelling StructureEntry:

After pre-screening process, the ICSD entries can be labelled now:

`disorder_label(entry, site_tolerance = 1e-4, vac_tolerance = 1e-2, occ_tolerance = 1.0, frac_tolerance = 1e-4,)` :

`site_tolerance`:

Determine if two sites are at the same position, in which case they will be combined to a single disordered site, defalut as 1e-4. For example: If the distance between two sites ≤ site_tolerance, the sites are regarded as in the same position.

`vac_tolerance:`

Determine if vacancy component exist for each single/multi-occupancy site, in which case the ‘VAC’ will present in the final disorder label, defalut as 1e-2. For example:

vacancy component = 1.0 - (sum of occupancy) ≥ vac_tolerance, the site are regarded as containing vacancy. Otherwise, the vacancy component will be ignored (e.g occ_sum = 1.0-0.995 = 0.005 ≤ 0.01, the vacancy component will be ignored and not recorded in disorder label).

`occ_tolerance`:

If sum of occupancy of a site is between 1 and occupancy_tolerance, it will be scaled down to 1. If total occupancy > occ_tolerance, the CollectionCode and sites will be recorded in `occ_err_sites`.  
**(notes: Entries that contain occupancy > occ_tolerance will never be passed into `is_positional_disorder` and `intersect_orb` for further processing. To avoid pymatgen complaint ERROR when processing positional disorder, it is strongly advised that a consistent `occ_tolerance` be used for both pre-screening and labeling.)**

`frac_tolerance`:

Determine the rounding precision for all algebraic operation involved in disorder_label, default as 1e-4. (note: It is recommended the rounding precision **be kept at 1e-4** or higher(1e-5), and use a tolerance in the 1e-x format, **avoiding coefficients other than 1** (e.g., avoid using 2e-4)

# 4. Post-processing:

After running `ICSD_filter.py` , following documents will be returned:

- A Processed ICSD dataframe, including the new columns:
  - Type of disorder (bool value): `is_disorder`, `is_sub_disorder` (substitutional), `is_vac_disorder` (vacancy), `is_positional_disorder`.
  - disorderd sites list:

    `disordered_label_list`(optional, not returned): e.g.  [[Co1, Mn1], [Ni1, Mn2]], it means the entry contains two disordered sites (or disordered Wyckoff positions). The first site is co-occupied with Co1 and Mn1, and the second site is co-occupied with Ni1 and Mn2.

    `disordered_valence_list` : the only difference with `disordered_label_list` is that sites co-occupied with **same element but in different oxidation number**, will be recorded as [Fe2+, Fe3+]. Sites with different elements remain as [Ni1, Mn2]].

  - `disorder_label`:

    Assembles a disorder_label as `{wyckoff_set_std}_{space_group_number}*_*{element_seq}`
    (e.g., 'j_i2_h_a_225_A_2B_{A+B}_C') for unique disorder structural identification.

    for more details, reference [3.2.1 Deduplicate ICSD](https://www.notion.so/3-2-1-Deduplicate-ICSD-22c368b0466180a5aa78ffa754f2f1b0?pvs=21)

  - `degree_of_mixing`:
    the **weighted average** of the **Shannon entropy** for all **co-occupied sites** (substitutional & vacancy-disordered sites) in the structure, with a sign delta:

                                                           Δ = (XA - XZ) / |XA - XZ|
    The site with the highest contribution to total mixing value is chosen as the representative.
    Δ is then derived from the two most extreme components (minimum XA and maximum XZ occupancies) of this representative site.

    for more details reference: [3.2.1 Deduplicate ICSD](https://www.notion.so/3-2-1-Deduplicate-ICSD-22c368b0466180a5aa78ffa754f2f1b0?pvs=21)

  - `fraction_of_disordered_sites`: number of co-occupied (S and V-type) sites / number of all sites
  - `wyckoff_set_std`: wyckoff sets, standardized Wyckoff set by converting the original wyckoff sets (under different space group setting) into a canonical format.
- Outliers documents:
  - `wyck_float_warn.csv`: triggered when number of generated coordinates > multiplicity claim on this site. This warns of inaccuracy of wyckoff position that has potential sever **floating-point error**. Conversely, if the number of generated coordinates < multiplicity, it is typically not a concern, as this usually indicates the coordinates lies on a special position (toward higher symmetry).
  - `occ_err_sites.csv`: sum of occupancy in a site that > `occ_tolerance`
  - `equivalent_but_distinct_sites.csv`: sites that are equivalent (the wyckoff positions that generate same sets of positions), but the coordinate of wyckoff positions are different.
  - `same_valence_sites.csv`: sites that are co-occupied with same elements with same oxidation number (this type of disorder is invalid and is automatically merged, the degree of mixing will also not be calculated).
  - `intersect_orb.csv`: Recording positional disordered sites. Two sites are intersect if distance between sites < threshold, refer to: [Exploration of positional disorder](https://www.notion.so/Exploration-of-positional-disorder-277368b04661808f9f59e25207e3a437?pvs=21)
  - `intersect_orb_error.json`: triggered when positional disorder related codes return error. Possible reasons: 1. contain elements that cannot define radius (noble gas, radioactive, etc); 2. Pymatgen Error(normally the occupancy>occ_tolerance is detect). 3. Pymatgen error: non-standard records in CIF. This **will remain empty** if consistent occ_tolerance are used in prescreeing and labelling.
  - `orb_none_entry.json`: The code failed to process wyckoff position based on symmetry operation recorded in CIF. **Normally, this will remain empty**. However, this might be useful for the user-generated CIF (e.g from generative model instead of ICSD).
  - `main_err.json`: Normally empty. The `disorder_label` fail to process `StructureEntry`. Contact me.


After running ICSD_filter.py, to generate a more clean ICSD database, the CollectionCode from following documents can be removed from ICSD if you trust the results:

- `wyck_float_warn.csv` : suggests a poor XRD refinement quality. The refined coordinates in CIF being shifted from their standard Wyckoff values.
- `occ_err_sites.csv`
- `equivalent_but_distinct_sites.csv` : usually contains floating-point error/
- `intersect_orb_error.json`
