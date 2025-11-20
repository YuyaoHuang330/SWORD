#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
from matplotlib import pyplot as plt

# ICSD = pd.read_csv('/home/users/jhwang/database/ICSD/ICSD2024_summary_2024.2_v5.3.0.csv')

icsd = pd.read_csv('deduplicated_multi_final.csv')
print(len(icsd))

allowed_elements = {"Ag", "Bi", "Ge", "Al", "Na", "Co", "Cu", "Ni",
                    "S", "Sb", "Se", "Sn", "Te", "Fe", "Ti"}

def get_elements(composition):
    elements = re.findall(r'[A-Z][a-z]?', composition)
    return set(elements)

icsd['elements_set'] = icsd['StructuredFormula'].apply(get_elements)
icsd = icsd[icsd['elements_set'].apply(lambda x: x.issubset(allowed_elements))]

def label_structure(elements):
    return len(elements)

icsd['nele'] = icsd['elements_set'].apply(label_structure)
icsd["nele"] = icsd["nele"].apply(lambda x: "4+" if x > 4 else x)

icsd = icsd[icsd['nele'].isin([1, 2, 3, 4, '4+'])]

icsd = icsd[["CollectionCode", "StructuredFormula", "elements_set", "is_disorder", "nele", "disorder_label"]]

icsd.to_csv("allowed_structures.csv", index=False)
print(icsd)

pivot_table = (
    icsd.pivot_table(
        index="nele",
        columns="is_disorder",
        aggfunc="size",
        fill_value=0
    )
)

pivot_table.columns = ["Ordered", "Disordered"]  # False/True -> Ordered/Disordered

print(pivot_table)
