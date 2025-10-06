import ieugwaspy as igp
import pandas as pd
from scipy.stats import norm
import numpy as np

from verifytoken import get_opengwas_jwt

def available_outcomes():

    outcomes = igp.gwasinfo()
    
    outcomes_list = [
        {"id": study_id, **meta_data}
        for study_id, meta_data in outcomes.items()
    ]

    outcomes = pd.DataFrame(outcomes_list)
    outcomes.fillna("NA", inplace=True)
    
    columns_order = [
        "id", "trait", "coverage", "ncase", "group_name", "year", "mr", "author", "sex",
        "qc_prior_to_upload", "pmid", "priority", "population", "unit", "nsnp", "sample_size",
        "build", "ncontrol", "covariates", "subcategory", "category", "ontology", "doi", "note",
        "consortium", "study_design", "sd"
    ]

    outcomes = outcomes[columns_order]
    outcomes = outcomes.sort_values(by=["id"], ascending=True)

    return outcomes

def extract_outcome_data(snps, outcomes, proxies=True, rsq=0.8, align_alleles=1,
                         palindromes=1, maf_threshold=0.3, splitsize=10000, proxy_splitsize=500):
    
    outcomes = igp.backwards.legacy_ids(list(set(outcomes)))
    snps = list(set(snps))

    firstpass = extract_outcome_data_internal(
        snps, outcomes, proxies=False,
        rsq=rsq,
        align_alleles=align_alleles,
        palindromes=palindromes,
        maf_threshold=maf_threshold
    )

    if proxies:
        for outcome in outcomes:
            if firstpass is None:
                missed_snps = snps
            else:
                missed_snps = [snp for snp in snps if snp not in firstpass[firstpass["id.outcome"] == outcome]["SNP"].tolist()]

            if missed_snps:
                print(f"Finding proxies for {len(missed_snps)} SNPs in outcome {outcome}")
                temp_df = extract_outcome_data_internal(
                    missed_snps, [outcome], proxies=True,
                    rsq=rsq,
                    align_alleles=align_alleles,
                    palindromes=palindromes,
                    maf_threshold=maf_threshold
                )
                if temp_df is not None:
                    firstpass = pd.concat([firstpass, temp_df], ignore_index=True)

    firstpass.fillna("NA", inplace=True)
    firstpass = firstpass.sort_values(by=["SNP"], ascending=True)
    return firstpass

def extract_outcome_data_internal(snps, outcomes, proxies=True, rsq=0.8, align_alleles=1,
                                  palindromes=1, maf_threshold=0.3, splitsize=10000):

    snps = list(set(snps))
    outcomes = list(set(outcomes))
    print(f"Extracting data for {len(snps)} SNP(s) from {len(outcomes)} GWAS(s)")

    if not isinstance(proxies, bool):
        raise ValueError("'proxies' argument should be True or False")

    if(proxies == False):
        proxies = 0
    elif(proxies == True):
        proxies = 1

    if (len(snps) < splitsize and len(outcomes) < splitsize):
        d = igp.associations(
            variant=snps,
            id=outcomes,
            proxies=proxies,
            r2=rsq,
            align_alleles=align_alleles,
            palindromes=palindromes,
            maf_threshold=maf_threshold
        )
        if isinstance(d, list):
            d = pd.DataFrame(d)
        elif not isinstance(d, pd.DataFrame):
            d = pd.DataFrame()


    elif len(snps) > len(outcomes):
        n = len(snps)
        splits = pd.DataFrame({"snps": snps, "chunk_id": [i // splitsize for i in range(n)]})
        d_list = []
        for i, outcome in enumerate(outcomes):
            print(f"{i + 1} of {len(outcomes)} outcomes")
            for chunk_id, chunk in splits.groupby("chunk_id"):
                print(f" [>] {chunk_id + 1} of {splits['chunk_id'].max() + 1} chunks")
                out = igp.associations(
                    variant=chunk["snps"].tolist(),
                    id=outcome,
                    proxies=proxies,
                    r2=rsq,
                    align_alleles=align_alleles,
                    palindromes=palindromes,
                    maf_threshold=maf_threshold
                )
                if isinstance(out, pd.DataFrame):
                    d_list.append(out)
        d = pd.concat(d_list, ignore_index=True) if d_list else pd.DataFrame()

    else:
        n = len(outcomes)
        splits = pd.DataFrame({"outcomes": outcomes, "chunk_id": [i // splitsize for i in range(n)]})
        d_list = []
        for i, snp in enumerate(snps):
            print(f"{i + 1} of {len(snps)} SNPs")
            for chunk_id, chunk in splits.groupby("chunk_id"):
                print(f" [>] {chunk_id + 1} of {splits['chunk_id'].max() + 1} chunks")
                out = igp.associations(
                    variant=[snp],
                    id=chunk["outcomes"].tolist(),
                    proxies=proxies,
                    r2=rsq,
                    align_alleles=align_alleles,
                    palindromes=palindromes,
                    maf_threshold=maf_threshold
                )
                if isinstance(out, pd.DataFrame):
                    d_list.append(out)
        d = pd.concat(d_list, ignore_index=True) if d_list else pd.DataFrame()

    if d.empty:
        print("None of the requested SNPs were available in the specified GWASs.")
        return None

    d = format_d(d)
    if not d.empty:
        d["data_source.outcome"] = "igd"
        d = d.sort_values(by=["SNP"], ascending=True)
        return d

    return None

def get_se(eff, pval):
    return abs(eff) / abs(norm.ppf(pval / 2))

def format_d(d):
    d1 = pd.DataFrame({
        "SNP": d["rsid"].astype(str),
        "chr": d["chr"].astype(str),
        "pos": d["position"].astype(str),
        "beta.outcome": pd.to_numeric(d["beta"], errors="coerce"),
        "se.outcome": pd.to_numeric(d["se"], errors="coerce"),
        "samplesize.outcome": pd.to_numeric(d["n"], errors="coerce"),
        "pval.outcome": pd.to_numeric(d["p"], errors="coerce"),
        "eaf.outcome": pd.to_numeric(d["eaf"], errors="coerce"),
        "effect_allele.outcome": d["ea"].astype(str),
        "other_allele.outcome": d["nea"].astype(str),
        "outcome": d["trait"].astype(str),
        "id.outcome": d["id"].astype(str)
    })
    
    if "proxy" in d.columns:
        p = pd.DataFrame({
            "proxy.outcome": d["proxy"],
            "target_snp.outcome": d["target_snp"],
            "proxy_snp.outcome": d["proxy_snp"],
            "target_a1.outcome": d["target_a1"],
            "target_a2.outcome": d["target_a2"],
            "proxy_a1.outcome": d["proxy_a1"],
            "proxy_a2.outcome": d["proxy_a2"]
        })
        d = pd.concat([d1, p], axis=1)      
        d = (d.groupby("outcome", group_keys=False).apply(lambda x: x.drop_duplicates(subset=["proxy_snp.outcome"])))

    else:
        d = d1
    
    if len(d) == 0:
        print("No matches")
        return d
    
    d["originalname.outcome"] = d["outcome"]
    d["outcome.deprecated"] = d.apply(lambda row: f"{row['outcome']} || {row.get('consortium.outcome', '')} || {row.get('year.outcome', '')}", axis=1)
    d["outcome"] = d.apply(lambda row: f"{row['outcome']} || id:{row['id.outcome']}", axis=1)
    
    d = d.dropna(subset=["beta.outcome", "pval.outcome"], how="all")
    
    index = d["se.outcome"].isna() | (d["se.outcome"] == 0) & (~d["beta.outcome"].isna() & ~d["pval.outcome"].isna())
    if index.any():
        d.loc[index, "se.outcome"] = get_se(d.loc[index, "beta.outcome"], d.loc[index, "pval.outcome"])
    
    d = cleanup_outcome_data(d)
    
    mrcols = ["beta.outcome", "se.outcome", "effect_allele.outcome"]
    d["mr_keep.outcome"] = d[mrcols].notna().all(axis=1)
    
    if not d["mr_keep.outcome"].all():
        missing_snps = d.loc[~d["mr_keep.outcome"], "SNP"].tolist()
        print("Warning: The following SNP(s) are missing required information for the MR tests and will be excluded\n" + "\n".join(missing_snps))
    
    if not d["mr_keep.outcome"].any():
        print("Warning: None of the provided SNPs can be used for MR analysis, they are missing required information.")
    
    return d

def cleanup_outcome_data(d):
    d.loc[d["se.outcome"] <= 0, "se.outcome"] = pd.NA
    d.loc[(d["eaf.outcome"] <= 0) | (d["eaf.outcome"] >= 1), "eaf.outcome"] = pd.NA
    d.loc[d["beta.outcome"] == -9, "beta.outcome"] = pd.NA
    return d