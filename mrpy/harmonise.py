import pandas as pd
import numpy as np

def harmonise_cleanup_variables(df):
    df["beta.exposure"] = pd.to_numeric(df["beta.exposure"], errors="coerce")
    df["beta.outcome"] = pd.to_numeric(df["beta.outcome"], errors="coerce")
    df["eaf.exposure"] = pd.to_numeric(df["eaf.exposure"], errors="coerce")
    df["eaf.outcome"] = pd.to_numeric(df["eaf.outcome"], errors="coerce")

    for col in ["effect_allele.exposure", "other_allele.exposure",
                "effect_allele.outcome", "other_allele.outcome"]:
        df[col] = df[col].str.upper()

    df["other_allele.outcome"] = df["other_allele.outcome"].replace("", np.nan)

    return df

def harmonise_make_snp_effects_positive(df):
    negative_effects = df["beta.exposure"] < 0
    df.loc[negative_effects, "beta.exposure"] *= -1
    df.loc[negative_effects, "eaf.exposure"] = 1 - df.loc[negative_effects, "eaf.exposure"]
    df.loc[negative_effects, ["effect_allele.exposure", "other_allele.exposure"]] = \
        df.loc[negative_effects, ["other_allele.exposure", "effect_allele.exposure"]].values
    return df

def check_palindromic(a1, a2):
    return ((a1 == "T") & (a2 == "A")) | ((a1 == "A") & (a2 == "T")) | ((a1 == "G") & (a2 == "C")) | ((a1 == "C") & (a2 == "G"))

def flip_alleles(allele):
    flip_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return allele.map(flip_map)

def recode_indels_22(A1, A2, B1, B2):
    ncA1 = A1.str.len()
    ncA2 = A2.str.len()
    ncB1 = B1.str.len()
    ncB2 = B2.str.len()

    i1 = (ncA1 > ncA2) & (B1 == "I") & (B2 == "D")
    B1.loc[i1] = A1.loc[i1]
    B2.loc[i1] = A2.loc[i1]

    i1 = (ncA1 < ncA2) & (B1 == "I") & (B2 == "D")
    B1.loc[i1] = A2.loc[i1]
    B2.loc[i1] = A1.loc[i1]

    i1 = (ncA1 > ncA2) & (B1 == "D") & (B2 == "I")
    B1.loc[i1] = A2.loc[i1]
    B2.loc[i1] = A1.loc[i1]

    i1 = (ncA1 < ncA2) & (B1 == "D") & (B2 == "I")
    B1.loc[i1] = A1.loc[i1]
    B2.loc[i1] = A2.loc[i1]

    i1 = (ncB1 > ncB2) & (A1 == "I") & (A2 == "D")
    A1.loc[i1] = B1.loc[i1]
    A2.loc[i1] = B2.loc[i1]

    i1 = (ncB1 < ncB2) & (A1 == "I") & (A2 == "D")
    A2.loc[i1] = B1.loc[i1]
    A1.loc[i1] = B2.loc[i1]

    i1 = (ncB1 > ncB2) & (A1 == "D") & (A2 == "I")
    A2.loc[i1] = B1.loc[i1]
    A1.loc[i1] = B2.loc[i1]

    i1 = (ncB1 < ncB2) & (A1 == "D") & (A2 == "I")
    A1.loc[i1] = B1.loc[i1]
    A2.loc[i1] = B2.loc[i1]

    keep = pd.Series([True] * len(A1))

    keep.loc[(ncA1 > 1) & (ncA1 == ncA2) & ((B1 == "D") | (B1 == "I"))] = False
    keep.loc[(ncB1 > 1) & (ncB1 == ncB2) & ((A1 == "D") | (A1 == "I"))] = False
    keep.loc[A1 == A2] = False
    keep.loc[B1 == B2] = False

    return pd.DataFrame({"A1": A1, "A2": A2, "B1": B1, "B2": B2, "keep": keep})

def recode_indels_21(A1, A2, B1):
    ncA1 = A1.str.len()
    ncA2 = A2.str.len()

    B2 = pd.Series([None] * len(B1), dtype="object")

    i1 = (ncA1 > ncA2) & (B1 == "I")
    B1.loc[i1] = A1.loc[i1]
    B2.loc[i1] = A2.loc[i1]

    i1 = (ncA1 < ncA2) & (B1 == "I")
    B1.loc[i1] = A2.loc[i1]
    B2.loc[i1] = A1.loc[i1]

    i1 = (ncA1 > ncA2) & (B1 == "D")
    B1.loc[i1] = A2.loc[i1]
    B2.loc[i1] = A1.loc[i1]

    i1 = (ncA1 < ncA2) & (B1 == "D")
    B1.loc[i1] = A1.loc[i1]
    B2.loc[i1] = A2.loc[i1]

    keep = pd.Series([True] * len(A1))

    keep.loc[(A1 == "I") & (A2 == "D")] = False
    keep.loc[(A1 == "D") & (A2 == "I")] = False
    keep.loc[(ncA1 > 1) & (ncA1 == ncA2) & ((B1 == "D") | (B1 == "I"))] = False
    keep.loc[A1 == A2] = False

    return pd.DataFrame({"A1": A1, "A2": A2, "B1": B1, "B2": B2, "keep": keep})

def recode_indels_12(A1, B1, B2):
    ncB1 = B1.str.len()
    ncB2 = B2.str.len()

    A2 = pd.Series([None] * len(A1), dtype="object")

    i1 = (ncB1 > ncB2) & (A1 == "I")
    A1.loc[i1] = B1.loc[i1]
    A2.loc[i1] = B2.loc[i1]

    i1 = (ncB1 < ncB2) & (A1 == "I")
    A2.loc[i1] = B1.loc[i1]
    A1.loc[i1] = B2.loc[i1]

    i1 = (ncB1 > ncB2) & (A1 == "D")
    A2.loc[i1] = B1.loc[i1]
    A1.loc[i1] = B2.loc[i1]

    i1 = (ncB1 < ncB2) & (A1 == "D")
    A1.loc[i1] = B1.loc[i1]
    A2.loc[i1] = B2.loc[i1]

    keep = pd.Series([True] * len(A1))

    keep.loc[(B1 == "I") & (B2 == "D")] = False
    keep.loc[(B1 == "D") & (B2 == "I")] = False
    keep.loc[(ncB1 > 1) & (ncB1 == ncB2) & ((A1 == "D") | (A1 == "I"))] = False
    keep.loc[B1 == B2] = False

    return pd.DataFrame({"A1": A1, "A2": A2, "B1": B1, "B2": B2, "keep": keep})

def harmonise_22(SNP, A1, A2, B1, B2, betaA, betaB, fA, fB, tolerance, action):
    if len(SNP) == 0:
        return pd.DataFrame()

    jlog = {"alleles": "2-2"}

    indel_index = (A1.str.len() > 1) | (A2.str.len() > 1) | (A1 == "D") | (A1 == "I")
    temp = recode_indels_22(A1[indel_index], A2[indel_index], B1[indel_index], B2[indel_index])

    A1[indel_index] = temp["A1"].values
    A2[indel_index] = temp["A2"].values
    B1[indel_index] = temp["B1"].values
    B2[indel_index] = temp["B2"].values

    status1 = (A1 == B1) & (A2 == B2)
    to_swap = (A1 == B2) & (A2 == B1)
    jlog["switched_alleles"] = np.sum(to_swap)

    Btemp = B1[to_swap]
    B1[to_swap] = B2[to_swap]
    B2[to_swap] = Btemp
    betaB[to_swap] *= -1
    fB[to_swap] = 1 - fB[to_swap]

    status1 = (A1 == B1) & (A2 == B2)
    palindromic = check_palindromic(A1, A2)

    i = ~palindromic & ~status1
    B1[i] = flip_alleles(B1[i])
    B2[i] = flip_alleles(B2[i])
    status1 = (A1 == B1) & (A2 == B2)
    jlog["flipped_alleles_basic"] = np.sum(i)

    i = ~palindromic & ~status1
    to_swap = (A1 == B2) & (A2 == B1)
    Btemp = B1[to_swap]
    B1[to_swap] = B2[to_swap]
    B2[to_swap] = Btemp
    betaB[to_swap] *= -1
    fB[to_swap] = 1 - fB[to_swap]

    status1 = (A1 == B1) & (A2 == B2)
    remove = ~status1
    remove[indel_index] = ~temp["keep"].values

    minf, maxf = 0.5 - tolerance, 0.5 + tolerance
    tempfA = fA.fillna(0.5)
    tempfB = fB.fillna(0.5)
    ambiguousA = (tempfA > minf) & (tempfA < maxf)
    ambiguousB = (tempfB > minf) & (tempfB < maxf)

    if action == 2:
        status2 = ((tempfA < 0.5) & (tempfB > 0.5)) | ((tempfA > 0.5) & (tempfB < 0.5))
        to_swap = status2 & palindromic & ~remove
        betaB[to_swap] *= -1
        fB[to_swap] = 1 - fB[to_swap]
        jlog["flipped_alleles_palindrome"] = np.sum(to_swap)
    else:
        jlog["flipped_alleles_palindrome"] = 0

    d = pd.DataFrame({
        "SNP": SNP,
        "effect_allele.exposure": A1,
        "other_allele.exposure": A2,
        "effect_allele.outcome": B1,
        "other_allele.outcome": B2,
        "beta.exposure": betaA,
        "beta.outcome": betaB,
        "eaf.exposure": fA,
        "eaf.outcome": fB,
        "remove": remove,
        "palindromic": palindromic,
        "ambiguous": (ambiguousA | ambiguousB) & palindromic
    })

    d.attrs["log"] = jlog

    return d

def harmonise_21(SNP, A1, A2, B1, betaA, betaB, fA, fB, tolerance, action):
    if len(SNP) == 0:
        return pd.DataFrame()

    jlog = {"alleles": "2-1"}

    n = len(A1)
    B2 = pd.Series([None] * n)
    ambiguous = pd.Series([False] * n)
    palindromic = check_palindromic(A1, A2)
    remove = palindromic

    indel_index = (A1.str.len() > 1) | (A2.str.len() > 1) | (A1 == "D") | (A1 == "I")
    temp = recode_indels_21(A1[indel_index], A2[indel_index], B1[indel_index])

    A1[indel_index] = temp["A1"].values
    A2[indel_index] = temp["A2"].values
    B1[indel_index] = temp["B1"].values
    B2[indel_index] = temp["B2"].values
    remove[indel_index] = ~temp["keep"].values

    status1 = A1 == B1
    minf, maxf = 0.5 - tolerance, 0.5 + tolerance

    tempfA = fA.fillna(0.5)
    tempfB = fB.fillna(0.5)

    freq_similar1 = ((tempfA < minf) & (tempfB < minf)) | ((tempfA > maxf) & (tempfB > maxf))
    ambiguous[status1 & ~freq_similar1] = True

    B2[status1] = A2[status1]

    to_swap = A2 == B1
    jlog["switched_alleles"] = np.sum(to_swap)
    freq_similar2 = ((tempfA < minf) & (tempfB > maxf)) | ((tempfA > maxf) & (tempfB < minf))

    ambiguous[to_swap & ~freq_similar2] = True
    B2[to_swap] = B1[to_swap]
    B1[to_swap] = A1[to_swap]
    betaB[to_swap] *= -1
    fB[to_swap] = 1 - fB[to_swap]

    to_flip = (A1 != B1) & (A2 != B1)
    jlog["flipped_alleles_no_oa"] = np.sum(to_flip)

    ambiguous[to_flip] = True

    B1[to_flip] = flip_alleles(B1[to_flip])
    status1 = A1 == B1
    B2[status1] = A2[status1]

    to_swap = A2 == B1
    B2[to_swap] = B1[to_swap]
    B1[to_swap] = A1[to_swap]
    betaB[to_swap] *= -1
    fB[to_swap] = 1 - fB[to_swap]

    d = pd.DataFrame({
        "SNP": SNP,
        "effect_allele.exposure": A1,
        "other_allele.exposure": A2,
        "effect_allele.outcome": B1,
        "other_allele.outcome": B2,
        "beta.exposure": betaA,
        "beta.outcome": betaB,
        "eaf.exposure": fA,
        "eaf.outcome": fB,
        "remove": remove,
        "palindromic": palindromic,
        "ambiguous": ambiguous | palindromic
    })

    d.attrs["log"] = jlog

    return d

def harmonise_12(SNP, A1, B1, B2, betaA, betaB, fA, fB, tolerance, action):
    if len(SNP) == 0:
        return pd.DataFrame()

    jlog = {"alleles": "1-2"}

    n = len(A1)
    A2 = pd.Series([None] * n)
    ambiguous = pd.Series([False] * n)
    palindromic = check_palindromic(B1, B2)
    remove = palindromic

    indel_index = (B1.str.len() > 1) | (B2.str.len() > 1) | (B1 == "D") | (B1 == "I")
    temp = recode_indels_12(A1[indel_index], B1[indel_index], B2[indel_index])

    A1[indel_index] = temp["A1"].values
    A2[indel_index] = temp["A2"].values
    B1[indel_index] = temp["B1"].values
    B2[indel_index] = temp["B2"].values
    remove[indel_index] = ~temp["keep"].values

    status1 = A1 == B1
    minf, maxf = 0.5 - tolerance, 0.5 + tolerance

    tempfA = fA.fillna(0.5)
    tempfB = fB.fillna(0.5)

    freq_similar1 = ((tempfA < minf) & (tempfB < minf)) | ((tempfA > maxf) & (tempfB > maxf))
    ambiguous[status1 & ~freq_similar1] = True

    A2[status1] = B2[status1]

    to_swap = A1 == B2
    jlog["switched_alleles"] = np.sum(to_swap)

    freq_similar2 = ((tempfA < minf) & (tempfB > maxf)) | ((tempfA > maxf) & (tempfB < maxf))
    ambiguous[to_swap & ~freq_similar2] = True
    A2[to_swap] = A1[to_swap]
    A1[to_swap] = B1[to_swap]
    betaA[to_swap] *= -1
    fA[to_swap] = 1 - fA[to_swap]

    to_flip = (A1 != B1) & (A1 != B2)
    jlog["flipped_alleles_no_oa"] = np.sum(to_flip)

    ambiguous[to_flip] = True

    A1[to_flip] = flip_alleles(A1[to_flip])
    status1 = A1 == B1
    A2[status1] = B2[status1]

    to_swap = B2 == A1
    B2[to_swap] = B1[to_swap]
    B1[to_swap] = A1[to_swap]
    betaB[to_swap] *= -1
    fB[to_swap] = 1 - fB[to_swap]

    d = pd.DataFrame({
        "SNP": SNP,
        "effect_allele.exposure": A1,
        "other_allele.exposure": A2,
        "effect_allele.outcome": B1,
        "other_allele.outcome": B2,
        "beta.exposure": betaA,
        "beta.outcome": betaB,
        "eaf.exposure": fA,
        "eaf.outcome": fB,
        "remove": remove,
        "palindromic": palindromic,
        "ambiguous": ambiguous | palindromic
    })

    d.attrs["log"] = jlog

    return d

def harmonise_11(SNP, A1, B1, betaA, betaB, fA, fB, tolerance, action):
    if len(SNP) == 0:
        return pd.DataFrame()

    jlog = {"alleles": "1-1"}

    n = len(A1)
    A2 = pd.Series([None] * n) 
    B2 = pd.Series([None] * n)
    ambiguous = pd.Series([False] * n)
    palindromic = False

    status1 = A1 == B1
    remove = ~status1

    minf, maxf = 0.5 - tolerance, 0.5 + tolerance

    tempfA = fA.fillna(0.5)
    tempfB = fB.fillna(0.5)

    freq_similar1 = ((tempfA < minf) & (tempfB < minf)) | ((tempfA > maxf) & (tempfB > maxf))
    ambiguous[status1 & ~freq_similar1] = True

    d = pd.DataFrame({
        "SNP": SNP,
        "effect_allele.exposure": A1,
        "other_allele.exposure": A2,
        "effect_allele.outcome": B1,
        "other_allele.outcome": B2,
        "beta.exposure": betaA,
        "beta.outcome": betaB,
        "eaf.exposure": fA,
        "eaf.outcome": fB,
        "remove": remove,
        "palindromic": palindromic,
        "ambiguous": ambiguous | palindromic
    })

    d.attrs["log"] = jlog

    return d

def check_required_columns(df, dataset_type):
    required_columns = [
        "SNP",
        f"beta.{dataset_type}",
        f"se.{dataset_type}",
        f"effect_allele.{dataset_type}",
        f"other_allele.{dataset_type}"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for {dataset_type}: {', '.join(missing_columns)}")

    print(f"All required columns for {dataset_type} are present.")

def harmonise(dat, tolerance=0.08, action=2):
    dat["orig_SNP"] = dat["SNP"]
    dat["SNP_index"] = dat.groupby("SNP").cumcount() + 1
    dat["SNP"] = dat["SNP"] + "_" + dat["SNP_index"].astype(str)

    SNP = dat["SNP"]
    A1 = dat["effect_allele.exposure"]
    A2 = dat["other_allele.exposure"]
    B1 = dat["effect_allele.outcome"]
    B2 = dat["other_allele.outcome"]
    betaA = dat["beta.exposure"]
    betaB = dat["beta.outcome"]
    fA = dat["eaf.exposure"]
    fB = dat["eaf.outcome"]

    i22 = A1.notna() & A2.notna() & B1.notna() & B2.notna()
    i21 = A1.notna() & A2.notna() & B1.notna() & B2.isna()
    i12 = A1.notna() & A2.isna() & B1.notna() & B2.notna()
    i11 = A1.notna() & A2.isna() & B1.notna() & B2.isna()

    d22 = harmonise_22(SNP[i22], A1[i22], A2[i22], B1[i22], B2[i22], betaA[i22], betaB[i22], fA[i22], fB[i22], tolerance, action)
    d21 = harmonise_21(SNP[i21], A1[i21], A2[i21], B1[i21], betaA[i21], betaB[i21], fA[i21], fB[i21], tolerance, action)
    d12 = harmonise_12(SNP[i12], A1[i12], B1[i12], B2[i12], betaA[i12], betaB[i12], fA[i12], fB[i12], tolerance, action)
    d11 = harmonise_11(SNP[i11], A1[i11], B1[i11], betaA[i11], betaB[i11], fA[i11], fB[i11], tolerance, action)

    d = pd.concat([d21, d22, d12, d11], ignore_index=True)

    d = d.merge(dat.drop(columns=["effect_allele.exposure", "other_allele.exposure", 
                                  "effect_allele.outcome", "other_allele.outcome", 
                                  "beta.exposure", "beta.outcome", "eaf.exposure", "eaf.outcome"]), 
                on="SNP", how="left")
    
    d["SNP"] = d["orig_SNP"]
    d = d.drop(columns=["orig_SNP"])
    d = d.sort_values(by=["id.outcome"])

    d["remove"] = d["remove"].astype(bool)
    d["ambiguous"] = d["ambiguous"].astype(bool)
    d["palindromic"] = d["palindromic"].astype(bool)

    d["mr_keep"] = True

    if action == 3:
        d["mr_keep"] = ~(d["palindromic"] | d["remove"] | d["ambiguous"])
        print("Removing the following SNPs for being palindromic:", d.loc[d["palindromic"], "SNP"].tolist())
        print("Removing the following SNPs for incompatible alleles:", d.loc[d["remove"], "SNP"].tolist())
        print("Removing the following SNPs for having incompatible allele frequencies:", d.loc[d["ambiguous"] & ~d["palindromic"], "SNP"].tolist())

    elif action == 2:
        d["mr_keep"] = ~(d["remove"] | d["ambiguous"])
        print("Removing the following SNPs for incompatible alleles:", d.loc[d["remove"], "SNP"].tolist())
        print("Removing the following SNPs for being palindromic with intermediate allele frequencies:", d.loc[d["ambiguous"], "SNP"].tolist())

    elif action == 1:
        d["mr_keep"] = ~d["remove"]
        print("Removing the following SNPs for incompatible alleles:", d.loc[d["remove"], "SNP"].tolist())


    return d


def harmonise_data(exposure_dat, outcome_dat, action=2):

    if not all(x in [1, 2, 3] for x in np.atleast_1d(action)):
        raise ValueError("Action must be one of [1, 2, 3].")
    
    check_required_columns(exposure_dat, "exposure")
    check_required_columns(outcome_dat, "outcome")

    res_tab = pd.merge(outcome_dat, exposure_dat, on="SNP", suffixes=("", "_exposure"))
    if "id.exposure" not in res_tab.columns:
        if "id.outcome_exposure" in res_tab.columns:
            res_tab["id.exposure"] = res_tab["id.outcome_exposure"]
        elif "id.outcome" in exposure_dat.columns:
            res_tab = res_tab.merge(
                exposure_dat[["SNP", "id.outcome"]].rename(columns={"id.outcome": "id.exposure"}),
                on="SNP", how="left"
            )
        else:
            raise ValueError("Cannot find 'id.exposure' in merged dataframe.")

    n_combinations = len(res_tab["id.outcome"].unique())

    if isinstance(action, int):
        action = [action] * n_combinations
    elif len(action) != n_combinations:
        raise ValueError("Action must be of length 1 or match the number of unique outcomes.")

    res_tab = harmonise_cleanup_variables(res_tab)

    unique_outcomes = res_tab["id.outcome"].unique()
    actions_df = pd.DataFrame({"id.outcome": unique_outcomes, "action": action})
    res_tab = res_tab.merge(actions_df, on="id.outcome", how="left")

    fix_tab = []
    mr_cols = ["beta.exposure", "beta.outcome", "se.exposure", "se.outcome"]

    for (_, group), current_action in zip(res_tab.groupby(["id.exposure", "id.outcome"]), action):
        print(f"Harmonising {group['exposure'].iloc[0]} ({group['id.exposure'].iloc[0]}) "
            f"and {group['outcome'].iloc[0]} ({group['id.outcome'].iloc[0]}) with action = {current_action}")

        group = harmonise(group, 0.08, current_action)

        candidate_variants = (res_tab["id.exposure"] == group["id.exposure"].iloc[0]).sum()
        variants_absent = candidate_variants - len(group)
        proxy_variants = group["proxy.outcome"].sum() if "proxy.outcome" in group.columns else 0

        total_variants = len(group)
        total_variants_for_mr = group[mr_cols].notna().all(axis=1).sum()

        log_data = {
            "candidate_variants": candidate_variants,
            "variants_absent_from_reference": variants_absent,
            "total_variants": total_variants,
            "total_variants_for_mr": total_variants_for_mr,
            "proxy_variants": proxy_variants
        }

        print(f"Log data for {group['id.exposure'].iloc[0]}-{group['id.outcome'].iloc[0]}: {log_data}")

        group["mr_keep"] = group["mr_keep"] & ~group[mr_cols].isnull().any(axis=1)
        fix_tab.append(group)

    columns_order = [
        "SNP", "effect_allele.exposure", "other_allele.exposure", "effect_allele.outcome", "other_allele.outcome",
        "beta.exposure", "beta.outcome", "eaf.exposure", "eaf.outcome", "remove", "palindromic", "ambiguous",
        "id.outcome", "chr", "pos", "se.outcome", "samplesize.outcome", "pval.outcome", "outcome",
        "originalname.outcome", "outcome.deprecated", "mr_keep.outcome", "data_source.outcome", "pval.exposure",
        "samplesize.exposure", "chr.exposure", "se.exposure", "pos.exposure", "id.exposure", "exposure",
        "mr_keep.exposure", "pval_origin.exposure", "data_source.exposure", "action", "SNP_index", "mr_keep"
    ]

    harmonised_data = pd.concat(fix_tab, ignore_index=True)
    for col in columns_order:
        if col not in harmonised_data.columns:
            harmonised_data[col] = pd.NA

    format_columns = [
        "beta.exposure", "se.exposure", "pval.exposure", "eaf.exposure",
        "beta.outcome", "se.outcome", "pval.outcome", "eaf.outcome"
    ]

    for col in format_columns:
        if col in harmonised_data.columns:
            harmonised_data[col] = harmonised_data[col].apply(
                lambda x: float(f"{x:.5g}") if pd.notna(x) else x
            )

    harmonised_data = harmonised_data[columns_order]
    harmonised_data = harmonised_data.sort_values(by=["SNP", "id.outcome"], ascending=True)

    return harmonised_data