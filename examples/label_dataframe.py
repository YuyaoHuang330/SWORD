from pathlib import Path

import pandas as pd

from sword import label_dataframe


INPUT_PATH = Path("path/to/database_dataframe.pkl")
OUTPUT_PATH = Path("sword_labelled_dataframe.pkl")


df = pd.read_pickle(INPUT_PATH)

labelled, anomalies = label_dataframe(
    df,
    cif_col="cif",
    id_col="material_id",
    sword_params={
        "parser_occ_tolerance": 1.05,
        "site_tolerance": 1e-4,
        "occ_tolerance": 1.0,
        "vac_tolerance": 1e-2,
        "frac_tolerance": 1e-4,
    },
    # Used only when family_info=True.
    family_info=False,
    family_params={
        "parser_occ_tolerance": 10.0,
        "fill_vacancy": False,
    },
)

labelled.to_pickle(OUTPUT_PATH)
print("labelled rows:", len(labelled))
print("saved to:", OUTPUT_PATH)
