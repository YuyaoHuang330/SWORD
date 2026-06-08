from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from sword import (
    ICSDDedupResult,
    dedupe_by_dom_projection,
    group_ICSD,
    label_icsd_dataframe,
    prescreen_icsd_dataframe,
)


ANOMALY_KEYS = (
    "label_errors",
    "wyck_float_warn",
    "occ_err_sites",
    "equivalent_but_distinct_sites",
    "same_valence_sites",
    "intersect_orb_errors",
)

PRESCREEN_PARAMS = {
    "parser_occ_tolerance": 1.05,
    "excluded_elements": ("He", "Ne", "Ar", "Kr", "Es"),
    "exclude_hydrogen": True,
}

SWORD_PARAMS = {
    "symprec": 1e-2,
    "angle_tolerance": 5.0,
    "parser_occ_tolerance": 1.05,
    "site_tolerance": 1e-4,
    "occ_tolerance": 1.0,
    "vac_tolerance": 1e-2,
    "frac_tolerance": 1e-4,
    "conventional_struct": True,
    "refine_struct": False,
}

CURATION_FILTERS = {
    "drop_hard_errors": True,
    "drop_positional_disorder": True,
    "drop_occupancy_orbit_error": True,
    "drop_wyckoff_float_warning": True,
    "drop_equivalent_sites_warning": True,
    "drop_same_valence_site_warning": False,
}


def chunks(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield start // chunk_size, df.iloc[start : start + chunk_size].copy()


def label_chunk(args):
    chunk_id, chunk, cif_col, id_col, mode, family_info, family_params = args
    labels, anomalies = label_icsd_dataframe(
        chunk,
        cif_col=cif_col,
        id_col=id_col,
        mode=mode,
        sword_params=SWORD_PARAMS,
        family_info=family_info,
        family_params=family_params,
    )
    return chunk_id, len(chunk), labels, anomalies


def concat_tables(tables, *, empty_columns=None):
    tables = [table for table in tables if table is not None and len(table) > 0]
    if tables:
        return pd.concat(tables, ignore_index=True)
    return pd.DataFrame(columns=empty_columns or [])


def merge_anomalies(anomaly_dicts, id_col):
    return {
        key: concat_tables([item.get(key) for item in anomaly_dicts], empty_columns=[id_col])
        for key in ANOMALY_KEYS
    }


def build_refined_results(label_results, *, id_col, dom_distance_tol):
    df = label_results.dropna(subset=["SWORD_label"]).copy()
    if df.empty:
        return df

    issues = df["processing_issue"].fillna("")
    remove = pd.Series(False, index=df.index)
    if CURATION_FILTERS["drop_hard_errors"]:
        remove |= issues.str.contains("structure_parse_failed|label_generation_failed", regex=True)
    if CURATION_FILTERS["drop_positional_disorder"]:
        remove |= df["is_positional_disorder"].fillna(False).astype(bool)
        remove |= issues.str.contains("positional_check_failed", regex=False)
    if CURATION_FILTERS["drop_occupancy_orbit_error"]:
        remove |= issues.str.contains("occupancy_orbit_error", regex=False)
    if CURATION_FILTERS["drop_wyckoff_float_warning"]:
        remove |= issues.str.contains("wyckoff_float_warning", regex=False)
    if CURATION_FILTERS["drop_equivalent_sites_warning"]:
        remove |= issues.str.contains("equivalent_sites_warning", regex=False)
    if CURATION_FILTERS["drop_same_valence_site_warning"]:
        remove |= issues.str.contains("same_valence_site_warning", regex=False)

    df = df[~remove].copy()
    keep_ids = []
    for _, group in df.groupby("SWORD_label", sort=False):
        ordered = group[~group["is_disorder"].fillna(False).astype(bool)]
        disorder = group[group["is_disorder"].fillna(False).astype(bool)]
        if not ordered.empty:
            keep_ids.append(ordered.sort_values(id_col).iloc[0][id_col])
        if not disorder.empty:
            keep_ids.extend(
                dedupe_by_dom_projection(
                    disorder,
                    id_col=id_col,
                    dom_col="degree_of_mixing",
                    elements_col="dom_site_elements",
                    occupancies_col="dom_site_occupancies",
                    dominant_element_col="dom_dominant_element",
                    tol=dom_distance_tol,
                )
            )
    return df[df[id_col].isin(keep_ids)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("sword_icsd_pipeline_output"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--cif-col", default="cif")
    parser.add_argument("--id-col", default="CollectionCode")
    parser.add_argument("--mode", choices=("from_collection_code", "from_pmg"), default="from_collection_code")
    parser.add_argument("--family-info", action="store_true")
    parser.add_argument("--dom-distance-tol", type=float, default=0.03)
    args = parser.parse_args()

    family_params = {
        "parser_occ_tolerance": 10.0,
        "fill_vacancy": False,
    }

    df = pd.read_pickle(args.input)
    screened, rejected = prescreen_icsd_dataframe(
        df,
        cif_col=args.cif_col,
        id_col=args.id_col,
        **PRESCREEN_PARAMS,
    )

    tasks = [
        (chunk_id, chunk, args.cif_col, args.id_col, args.mode, args.family_info, family_params)
        for chunk_id, chunk in chunks(screened, args.chunk_size)
    ]

    label_tables = []
    anomaly_dicts = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(label_chunk, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            chunk_id, n_rows, labels, anomalies = future.result()
            label_tables.append(labels)
            anomaly_dicts.append(anomalies)
            print(f"finished chunk {chunk_id} ({n_rows} rows); progress {done}/{len(futures)}", flush=True)

    label_results = concat_tables(label_tables)
    anomalies = merge_anomalies(anomaly_dicts, args.id_col)
    label_groups = group_ICSD(label_results.dropna(subset=["SWORD_label"]).copy(), sword_col="SWORD_label")
    refined = build_refined_results(
        label_results,
        id_col=args.id_col,
        dom_distance_tol=args.dom_distance_tol,
    )

    result = ICSDDedupResult(
        prescreen_rejected=rejected,
        label_results=label_results,
        anomalies=anomalies,
        label_groups=label_groups,
        refined=refined,
        metadata={
            "input": str(args.input),
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "mode": args.mode,
            "family_info": args.family_info,
            "family_params": family_params if args.family_info else {},
            "prescreen_params": PRESCREEN_PARAMS,
            "sword_params": SWORD_PARAMS,
            "curation_filters": CURATION_FILTERS,
            "dom_distance_tol": args.dom_distance_tol,
        },
    )
    result.save(args.output_dir)


if __name__ == "__main__":
    main()
