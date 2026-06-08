from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from sword import label_dataframe


SWORD_PARAMS = {
    "parser_occ_tolerance": 1.05,
    "site_tolerance": 1e-4,
    "occ_tolerance": 1.0,
    "vac_tolerance": 1e-2,
    "frac_tolerance": 1e-4,
}

FAMILY_PARAMS = {
    "parser_occ_tolerance": 10.0,
    "fill_vacancy": False,
}


def chunks(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield start // chunk_size, df.iloc[start : start + chunk_size].copy()


def label_chunk(args):
    chunk_id, chunk, cif_col, id_col, family_info = args
    labelled, anomalies = label_dataframe(
        chunk,
        cif_col=cif_col,
        id_col=id_col,
        sword_params=SWORD_PARAMS,
        family_info=family_info,
        family_params=FAMILY_PARAMS,
    )
    return chunk_id, len(chunk), labelled, anomalies


def concat_tables(tables):
    tables = [table for table in tables if table is not None and len(table) > 0]
    if tables:
        return pd.concat(tables, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("sword_labelled_dataframe.pkl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--cif-col", default="cif")
    parser.add_argument("--id-col", default="material_id")
    parser.add_argument("--family-info", action="store_true")
    args = parser.parse_args()

    df = pd.read_pickle(args.input)
    tasks = [
        (chunk_id, chunk, args.cif_col, args.id_col, args.family_info)
        for chunk_id, chunk in chunks(df, args.chunk_size)
    ]

    label_tables = []
    anomaly_tables = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(label_chunk, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            chunk_id, n_rows, labelled, anomalies = future.result()
            label_tables.append(labelled)
            for key, table in anomalies.items():
                anomaly_tables.setdefault(key, []).append(table)
            print(f"finished chunk {chunk_id} ({n_rows} rows); progress {done}/{len(futures)}", flush=True)

    labelled = concat_tables(label_tables)
    labelled.to_pickle(args.output)

    anomaly_dir = args.output.with_suffix("")
    anomaly_dir.mkdir(exist_ok=True)
    for key, tables in anomaly_tables.items():
        concat_tables(tables).to_csv(anomaly_dir / f"{key}.csv", index=False)

    print("labelled rows:", len(labelled))
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
