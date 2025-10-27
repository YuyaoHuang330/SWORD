#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

ICSD = pd.read_csv('/home/users/jhwang/database/ICSD/ICSD2024_summary_2024.2_v5.3.0.csv')

target = ICSD[ICSD['CollectionCode'] == 70589]
cif = target['cif'].values[0]

print(cif)
