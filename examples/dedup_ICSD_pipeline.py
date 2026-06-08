from pathlib import Path

import pandas as pd

from sword import run_icsd_dedup_pipeline


INPUT_PATH = Path("path/to/icsd_dataframe.pkl")
OUTPUT_DIR = Path("sword_icsd_pipeline_output")


df = pd.read_pickle(INPUT_PATH)

result = run_icsd_dedup_pipeline(
    df,
    cif_col="cif",
    id_col="CollectionCode",
    mode="from_collection_code",
    prescreen_params={
        "parser_occ_tolerance": 1.05,
        "excluded_elements": ("He", "Ne", "Ar", "Kr", "Es"),
        "exclude_hydrogen": True,
    },
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
    dom_distance_tol=0.03,
    output_dir=OUTPUT_DIR,
)

print("prescreen rejected:", len(result.prescreen_rejected))
print("label results:", len(result.label_results))
print("label groups:", len(result.label_groups))
print("refined results:", len(result.refined))
print("saved to:", OUTPUT_DIR)
