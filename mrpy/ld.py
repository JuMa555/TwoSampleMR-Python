import pandas as pd
import requests
from read_data import random_string
from verifytoken import get_opengwas_jwt

def clump_data(dat, clump_kb=10000, clump_r2=0.001, clump_p1=1, clump_p2=1, pop="EUR", bfile=None, plink_bin=None):
    if not isinstance(dat, pd.DataFrame):
        raise ValueError("Expecting data frame returned from format_data")

    if "pval.exposure" in dat.columns and "pval.outcome" in dat.columns:
        print("pval.exposure and pval.outcome columns present. Using pval.exposure for clumping.")
        pval_column = "pval.exposure"
    elif "pval.exposure" not in dat.columns and "pval.outcome" in dat.columns:
        print("pval.exposure column not present, using pval.outcome column for clumping.")
        pval_column = "pval.outcome"
    elif "pval.exposure" not in dat.columns:
        print("pval.exposure not present, setting clumping p-value to 0.99 for all variants")
        dat["pval.exposure"] = 0.99
        pval_column = "pval.exposure"
    else:
        pval_column = "pval.exposure"

    if "id.exposure" not in dat.columns:
        dat["id.exposure"] = random_string(1)

    clump_input = pd.DataFrame({
        "rsid": dat["SNP"],
        "pval": dat[pval_column],
        "id": dat["id.exposure"]
    })

    clump_payload = {
        "rsid": clump_input["rsid"].tolist(),
        "pval": clump_input["pval"].tolist(),
        "pthresh": clump_p1,
        "r2": clump_r2,
        "kb": clump_kb,
        "pop": pop
    }

    token = get_opengwas_jwt()
    headers = {"Authorization": f"Bearer {token}"}

    res = requests.post("https://api.opengwas.io/api/ld/clump", json=clump_payload, headers=headers)

    try:
        data = res.json()
    except Exception as e:
        raise ValueError(f"Invalid JSON response: {e}")

    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        out_snps = set(data)
        filtered = dat[dat["SNP"].astype(str).isin(out_snps)].copy()
        return filtered
    else:
        raise ValueError(f"Unexpected format from clumping API: {data}")


def ld_matrix(variants, with_alleles=True, pop="EUR", bfile=None, plink_bin=None):
    if len(variants) > 500 and bfile is None:
        raise ValueError("SNP list must be smaller than 500.")

    if bfile is not None:
        raise NotImplementedError("Local LD matrix computation using bfile and plink_bin is not yet implemented.")

    payload = {
        "rsid": variants,
        "pop": pop
    }

    token = get_opengwas_jwt()
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post("https://api.opengwas.io/api/ld/matrix", json=payload, headers=headers)

    res = response.json()

    if res is None or not res.get("matrix"):
        raise ValueError("None of the requested variants were found")

    matrix = res["matrix"]
    snplist = res["snplist"]

    matrix = pd.DataFrame(matrix, dtype=float)
    matrix.index = snplist
    matrix.columns = snplist

    if not with_alleles:
        rename_map = {s: s.split("_")[0] for s in snplist}
        matrix.rename(index=rename_map, columns=rename_map, inplace=True)

    found_rsids = [s.split("_")[0] for s in snplist]
    missing = [s for s in variants if s not in found_rsids]
    if missing:
        print("Warning: The following variants are not present in the LD reference panel:\n" + "\n".join(missing))

    order = {s: i for i, s in enumerate(variants)}
    sorted_indices = sorted(matrix.index, key=lambda x: order.get(x.split("_")[0] if "_" in x else x, -1))
    matrix = matrix.loc[sorted_indices, sorted_indices]

    return matrix