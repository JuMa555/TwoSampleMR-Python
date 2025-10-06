import pandas as pd
import numpy as np
from scipy.stats import norm
import re
import random
import string

def read_outcome_data(
    filename,
    snps=None,
    sep=" ",
    phenotype_col="Phenotype",
    snp_col="SNP",
    beta_col="beta",
    se_col="se",
    eaf_col="eaf",
    effect_allele_col="effect_allele",
    other_allele_col="other_allele",
    pval_col="pval",
    units_col="units",
    ncase_col="ncase",
    ncontrol_col="ncontrol",
    samplesize_col="samplesize",
    gene_col="gene",
    id_col="id",
    min_pval=1e-200,
    log_pval=False,
    chr_col="chr",
    pos_col="pos"
):
    outcome_dat = pd.read_csv(filename, sep=sep)

    outcome_dat = format_data(
        outcome_dat,
        type="outcome",
        snps=snps,
        phenotype_col=phenotype_col,
        snp_col=snp_col,
        beta_col=beta_col,
        se_col=se_col,
        eaf_col=eaf_col,
        effect_allele_col=effect_allele_col,
        other_allele_col=other_allele_col,
        pval_col=pval_col,
        units_col=units_col,
        ncase_col=ncase_col,
        ncontrol_col=ncontrol_col,
        samplesize_col=samplesize_col,
        gene_col=gene_col,
        id_col=id_col,
        min_pval=min_pval,
        log_pval=log_pval,
        chr_col=chr_col,
        pos_col=pos_col
    )

    outcome_dat["data_source.outcome"] = "textfile"
    return outcome_dat


def read_exposure_data(
    filename,
    clump=False,
    sep=" ",
    phenotype_col="Phenotype",
    snp_col="SNP",
    beta_col="beta",
    se_col="se",
    eaf_col="eaf",
    effect_allele_col="effect_allele",
    other_allele_col="other_allele",
    pval_col="pval",
    units_col="units",
    ncase_col="ncase",
    ncontrol_col="ncontrol",
    samplesize_col="samplesize",
    gene_col="gene",
    id_col="id",
    min_pval=1e-200,
    log_pval=False,
    chr_col="chr",
    pos_col="pos"
):
    from ld import clump_data
    exposure_dat = pd.read_csv(filename, sep=sep)

    exposure_dat = format_data(
        exposure_dat,
        type="exposure",
        snps=None,
        phenotype_col=phenotype_col,
        snp_col=snp_col,
        beta_col=beta_col,
        se_col=se_col,
        eaf_col=eaf_col,
        effect_allele_col=effect_allele_col,
        other_allele_col=other_allele_col,
        pval_col=pval_col,
        units_col=units_col,
        ncase_col=ncase_col,
        ncontrol_col=ncontrol_col,
        samplesize_col=samplesize_col,
        gene_col=gene_col,
        id_col=id_col,
        min_pval=min_pval,
        log_pval=log_pval,
        chr_col=chr_col,
        pos_col=pos_col
    )

    exposure_dat["data_source.exposure"] = "textfile"

    if clump:
        exposure_dat = clump_data(exposure_dat)

    return exposure_dat


def format_data(
    dat, type="exposure", snps=None, phenotype_col="Phenotype", snp_col="SNP",
    beta_col="beta", se_col="se", eaf_col="eaf", effect_allele_col="effect_allele",
    other_allele_col="other_allele", pval_col="pval", units_col="units",
    ncase_col="ncase", ncontrol_col="ncontrol", samplesize_col="samplesize",
    gene_col="gene", id_col="id", min_pval=1e-200, z_col="z", info_col="info",
    chr_col="chr", pos_col="pos", log_pval=False
):
    if not isinstance(dat, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    all_cols = [
        phenotype_col, snp_col, beta_col, se_col, eaf_col, effect_allele_col,
        other_allele_col, pval_col, units_col, ncase_col, ncontrol_col,
        samplesize_col, gene_col, id_col, z_col, info_col, chr_col, pos_col
    ]

    present_cols = [col for col in all_cols if col in dat.columns]
    if not present_cols:
        raise ValueError("None of the specified columns are present in the data.")

    dat = dat[present_cols]

    if snp_col not in dat.columns:
        raise ValueError("SNP column not found.")
    
    dat = dat.rename(columns={snp_col: "SNP"})
    dat.loc[:, "SNP"] = dat["SNP"].str.lower().str.replace(r"\s+", "", regex=True)

    dat = dat.dropna(subset=["SNP"])

    if snps is not None:
        dat = dat[dat["SNP"].isin(snps)]

    if phenotype_col not in dat.columns:
        dat[type] = type
    else:
        dat[type] = dat[phenotype_col]
        if phenotype_col != type:
            dat.drop(columns=[phenotype_col], inplace=True)

    if log_pval and pval_col in dat.columns:
        dat[pval_col] = 10 ** -dat[pval_col]

    dat = dat.groupby(type).apply(lambda x: x.drop_duplicates(subset=["SNP"])).reset_index(drop=True)

    mr_cols_required = ["SNP", beta_col, se_col, effect_allele_col]
    dat["mr_keep.outcome"] = all(col in dat.columns for col in mr_cols_required)

    mr_cols_desired = [other_allele_col, eaf_col]
    for col in mr_cols_desired:
        if col not in dat.columns:
            print(f"Warning: column {col} not present but helpful for harmonisation.")

    if beta_col in dat.columns:
        dat.rename(columns={beta_col: "beta.outcome"}, inplace=True)
        dat["beta.outcome"] = pd.to_numeric(dat["beta.outcome"], errors='coerce')
        dat.loc[~np.isfinite(dat["beta.outcome"]), "beta.outcome"] = np.nan

    if se_col in dat.columns:
        dat.rename(columns={se_col: "se.outcome"}, inplace=True)
        dat["se.outcome"] = pd.to_numeric(dat["se.outcome"], errors='coerce')
        invalid_se = (~np.isfinite(dat["se.outcome"])) | (dat["se.outcome"] <= 0)
        dat.loc[invalid_se, "se.outcome"] = np.nan

    if eaf_col in dat.columns:
        dat.rename(columns={eaf_col: "eaf.outcome"}, inplace=True)
        dat["eaf.outcome"] = pd.to_numeric(dat["eaf.outcome"], errors='coerce')
        invalid_eaf = (~np.isfinite(dat["eaf.outcome"])) | (dat["eaf.outcome"] <= 0) | (dat["eaf.outcome"] >= 1)
        dat.loc[invalid_eaf, "eaf.outcome"] = np.nan

    if effect_allele_col in dat.columns:
        dat.rename(columns={effect_allele_col: "effect_allele.outcome"}, inplace=True)
        valid_effect_alleles = lambda x: bool(re.match(r"^[ACTGDI]+$", str(x)))
        invalid_effect_alleles = ~dat["effect_allele.outcome"].apply(valid_effect_alleles)
        dat.loc[invalid_effect_alleles, "effect_allele.outcome"] = np.nan

    if other_allele_col in dat.columns:
        dat.rename(columns={other_allele_col: "other_allele.outcome"}, inplace=True)
        valid_other_alleles = lambda x: bool(re.match(r"^[ACTGDI]+$", str(x)))
        invalid_other_alleles = ~dat["other_allele.outcome"].apply(valid_other_alleles)
        dat.loc[invalid_other_alleles, "other_allele.outcome"] = np.nan

    if pval_col in dat.columns:
        dat = dat.rename(columns={pval_col: "pval.outcome"})
        dat["pval.outcome"] = pd.to_numeric(dat["pval.outcome"], errors="coerce")
        invalid = ~np.isfinite(dat["pval.outcome"]) | (dat["pval.outcome"] < 0) | (dat["pval.outcome"] > 1)
        dat.loc[invalid, "pval.outcome"] = np.nan
        dat["pval.outcome"] = dat["pval.outcome"].fillna(min_pval)
        dat["pval_origin.outcome"] = "reported"
        missing = dat["pval.outcome"].isna() & dat["beta.outcome"].notna() & dat["se.outcome"].notna()
        dat.loc[missing, "pval.outcome"] = 2 * norm.sf(abs(dat.loc[missing, "beta.outcome"] / dat.loc[missing, "se.outcome"]))
        dat.loc[missing, "pval_origin.outcome"] = "inferred"

    if beta_col in dat.columns and se_col in dat.columns and pval_col not in dat.columns:
        print("Inferring p-values")
        dat["pval.outcome"] = norm.sf(np.abs(dat[beta_col]) / dat[se_col]) * 2
        dat["pval_origin.outcome"] = "inferred"

    if ncase_col in dat.columns:
        dat.rename(columns={ncase_col: "ncase.outcome"}, inplace=True)
        if not pd.api.types.is_numeric_dtype(dat["ncase.outcome"]):
            print(f"{ncase_col} column is not numeric")
            dat["ncase.outcome"] = pd.to_numeric(dat["ncase.outcome"], errors='coerce')

    if ncontrol_col in dat.columns:
        dat.rename(columns={ncontrol_col: "ncontrol.outcome"}, inplace=True)
        if not pd.api.types.is_numeric_dtype(dat["ncontrol.outcome"]):
            print(f"{ncontrol_col} column is not numeric")
            dat["ncontrol.outcome"] = pd.to_numeric(dat["ncontrol.outcome"], errors='coerce')

    if samplesize_col in dat.columns:
        dat.rename(columns={samplesize_col: "samplesize.outcome"}, inplace=True)
        if not pd.api.types.is_numeric_dtype(dat["samplesize.outcome"]):
            print(f"{samplesize_col} column is not numeric")
            dat["samplesize.outcome"] = pd.to_numeric(dat["samplesize.outcome"], errors='coerce')

        if "ncontrol.outcome" in dat.columns and "ncase.outcome" in dat.columns:
            index = (
                dat["samplesize.outcome"].isna() &
                ~dat["ncase.outcome"].isna() &
                ~dat["ncontrol.outcome"].isna()
            )
            if index.any():
                print("Generating sample size from ncase and ncontrol")
                dat.loc[index, "samplesize.outcome"] = (
                    dat.loc[index, "ncase.outcome"] + dat.loc[index, "ncontrol.outcome"]
                )
    elif "ncontrol.outcome" in dat.columns and "ncase.outcome" in dat.columns:
        print("Generating sample size from ncase and ncontrol")
        dat["samplesize.outcome"] = dat["ncase.outcome"] + dat["ncontrol.outcome"]

    if gene_col in dat.columns:
        dat.rename(columns={gene_col: "gene.outcome"}, inplace=True)

    if info_col in dat.columns:
        dat.rename(columns={info_col: "info.outcome"}, inplace=True)

    if z_col in dat.columns:
        dat.rename(columns={z_col: "z.outcome"}, inplace=True)

    if chr_col in dat.columns:
        dat.rename(columns={chr_col: "chr.outcome"}, inplace=True)

    if pos_col in dat.columns:
        dat.rename(columns={pos_col: "pos.outcome"}, inplace=True)

    if units_col in dat.columns:
        dat.rename(columns={units_col: "units.outcome"}, inplace=True)
        temp_units = check_units(dat, type, "units.outcome")
        if temp_units.get("ph"):
            dat[type] = (
                f"{dat[type]} ({dat['units.outcome']})"
            )

    if id_col in dat.columns:
        dat.rename(columns={id_col: "id.outcome"}, inplace=True)
        dat["id.outcome"] = dat["id.outcome"].astype(str)
    else:
        create_ids(dat[type])

    if "mr_keep.outcome" in dat.columns:
        mrcols = ["SNP", "beta.outcome", "se.outcome", "effect_allele.outcome"]
        existing = [col for col in mrcols if col in dat.columns]
        dat["mr_keep.outcome"] = dat["mr_keep.outcome"] & dat[existing].notna().all(axis=1)
        if not dat["mr_keep.outcome"].all():
            missing = dat.loc[~dat["mr_keep.outcome"], "SNP"].tolist()
            print("Warning: The following SNP(s) are missing required information for the MR tests and will be excluded\n" + "\n".join(missing))

    if not dat["mr_keep.outcome"].any():
        print("Warning: None of the provided SNPs can be used for MR analysis, they are missing required information.")

    for col in ["SNP", "beta.outcome", "se.outcome", "effect_allele.outcome", "other_allele.outcome", "eaf.outcome"]:
        if col not in dat.columns:
            dat[col] = np.nan

    dat.columns = [c.replace(".outcome", f".{type}") for c in dat.columns]
    dat = dat.reset_index(drop=True)
    return dat


def check_units(data, id_col, unit_col):
    temp = data.groupby(id_col).apply(lambda group: {
        "ph": len(group[unit_col].unique()) > 1
    }).reset_index()

    for _, row in temp.iterrows():
        if row["ph"]:
            print(f"Warning: More than one type of unit specified for {row[id_col]}")
    
    return temp


def create_ids(x):
    unique_values = pd.Series(x).unique()
    random_strings = random_string(len(unique_values))
    id_mapping = dict(zip(unique_values, random_strings))
    return pd.Series(x).map(id_mapping).tolist()


def random_string(n=1, length=6):
    random_strings = []
    for _ in range(n):
        random_strings.append(''.join(random.choices(string.ascii_letters + string.digits, k=length)))
    return random_strings


def convert_outcome_to_exposure(outcome_dat: pd.DataFrame) -> pd.DataFrame:
    id_map = outcome_dat.drop_duplicates(subset="outcome")[["outcome", "id.outcome"]]

    exposure_dat = format_data(
        outcome_dat,
        beta_col="beta.outcome",
        se_col="se.outcome",
        pval_col="pval.outcome",
        phenotype_col="outcome",
        effect_allele_col="effect_allele.outcome",
        other_allele_col="other_allele.outcome",
        eaf_col="eaf.outcome",
        units_col="units.outcome"
    )

    exposure_dat = exposure_dat.merge(id_map, how="left", left_on="exposure", right_on="outcome")

    return exposure_dat

