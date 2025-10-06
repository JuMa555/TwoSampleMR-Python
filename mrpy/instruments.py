import pandas as pd
import ieugwaspy as igp
from verifytoken import get_opengwas_jwt
from read_data import format_data

def extract_instruments(outcomes, p1=5e-8, clump=1, p2=5e-8, r2=0.001, kb=10000, opengwas_jwt=None, force_server=False):
    if opengwas_jwt is None:
        opengwas_jwt = get_opengwas_jwt()
    
    outcomes = igp.backwards.legacy_ids(list(set(outcomes)))
    
    d = igp.tophits(outcomes, pval=p1, clump=clump, r2=r2, kb=kb, force_server=force_server)

    if isinstance(d, dict):
        df = pd.DataFrame({k: [v] for k, v in d.items()})
    else:
        df = pd.DataFrame(d)

    if df.empty:
        return None

    df["phenotype"] = df["trait"] + " || id:" + df["id"]
    
    df = format_data(
        df,
        type="exposure",
        snps=None,
        phenotype_col="phenotype",
        snp_col="rsid",
        chr_col="chr",
        pos_col="position",
        beta_col="beta",
        se_col="se",
        eaf_col="eaf",
        effect_allele_col="ea",
        other_allele_col="nea",
        pval_col="p",
        samplesize_col="n",
        min_pval=1e-200,
        id_col="id"
    )
    
    df["data_source.exposure"] = "igd"

    columns_order = [
        "pval.exposure", "samplesize.exposure", "chr.exposure", "se.exposure", "beta.exposure", "pos.exposure",
        "id.exposure", "SNP", "effect_allele.exposure", "other_allele.exposure", "eaf.exposure", "exposure",
        "mr_keep.exposure", "pval_origin.exposure", "data_source.exposure"
    ]

    df = df[columns_order]
    df = df.sort_values(by=["pval.exposure"], ascending=True)

    return df