#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wang Jianghai @NTU
Contact: jianghai001@e.ntu.edu.sg
Date: 2025-11-03
Description: 
"""
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd


# Deduplicate by selecting CollectionCode with highest absolute value of degree of mixing (would give only one representative for each structure group)
def dedupe_by_highest_DOM(deg_list, code_list):
    if deg_list is None or len(deg_list) == 0:
        return code_list[0] if code_list else np.nan

    arr = np.array(deg_list, dtype=float)

    if np.all(np.isnan(arr)):
        return code_list[0] if code_list else np.nan

    idx = int(np.nanargmax(np.abs(arr)))

    if code_list is None or idx >= len(code_list):
        return np.nan
    return code_list[idx]


# Deduplicate by removing CollectionCode with same degree of mixing (would give multiple representatives for each structure group)
def dedupe_by_same_DOM(deg_list, code_list):
    if deg_list is None or code_list is None:
        return code_list or []
    out = []
    seen = set()
    for d, c in zip(deg_list, code_list):
        key = ('__nan__' if pd.isna(d) else float(d))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


with open("ICSD_valid_dedup_Stol_1e-4_vtol_1e-2_occ_tol_1.05.pkl", "rb") as f:
    df_main = pd.read_pickle(f)

intersect_orb_sites = pd.read_csv("intersect_orb_sites.csv")["CollectionCode"]
equivalent_but_distinct_sites_data = pd.read_csv("equivalent_but_distinct_sites.csv")["CollectionCode"]
occ_err_sites = pd.read_csv("occ_err_sites.csv")["CollectionCode"]
same_valence_sites = pd.read_csv("same_valence_sites.csv")["CollectionCode"]
wyck_float_warning = pd.read_csv("wyck_float_warn.csv")["CollectionCode"]

forbidden_ids = pd.concat([
    pd.Series(intersect_orb_sites),
    pd.Series(equivalent_but_distinct_sites_data),
    pd.Series(occ_err_sites),
    pd.Series(same_valence_sites),
    pd.Series(wyck_float_warning),
    pd.Series(["148749"])
]).unique().tolist()

# normalize main id values to compare as strings trimmed
filtered = df_main[~df_main["CollectionCode"].isin(forbidden_ids)]
# filtered.to_csv("filtered_valid.csv", index=False)

ICSD_group = (
    filtered
    .groupby(['disorder_label'], as_index=False)
    .agg(
        n_total            = ('disorder_label', 'size'),
        CollectionCode_list= ('CollectionCode', list),
        StructuredFormula_list=('StructuredFormula', list),
        WyckoffSequence_lists = ('WyckoffSequence',list),
        # disorder_lists = ('disordered_list',list),
        degree_of_mixing_list=('degree_of_mixing', list),
        n_disorder         = ('is_disorder', 'sum'),
    )
)

ICSD_group['n_disorder'] = ICSD_group['n_disorder'].astype(int)

ICSD_group['selected_collection_code'] = ICSD_group.apply(
    lambda r: dedupe_by_same_DOM(r['degree_of_mixing_list'], r['CollectionCode_list']),
    axis=1
)

duplicated_codes = {c for v in ICSD_group['selected_collection_code'].dropna() for c in (v if isinstance(v, (list,tuple,set)) else [v])}
ICSD_deduplicated = filtered[filtered['CollectionCode'].isin(duplicated_codes)].copy()

ICSD_deduplicated.to_csv("deduplicated_multi_final.csv", index=False)

print(len(df_main), len(ICSD_deduplicated))
print(ICSD_deduplicated.columns)
