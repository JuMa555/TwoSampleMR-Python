import numpy as np
import pandas as pd
from mr import default_parameters, mr_method_list, method_map, mr_egger_regression

def mr_heterogeneity(dat, parameters=None, method_list=None):
    if parameters is None:
        parameters = default_parameters()

    if method_list is None:
        method_list = [
            m["obj"]
            for m in mr_method_list()
            if m["heterogeneity_test"] and m["use_by_default"]
        ]

    results = []

    grouped = dat.groupby(["id.exposure", "id.outcome"])

    for (id_exp, id_out), group in grouped:
        x = group[group["mr_keep"]].copy()

        if len(x) < 2:
            print(f"Not enough SNPs available for Heterogeneity analysis of '{id_exp}' on '{id_out}'")
            continue

        res_list = []
        for method_name in method_list:
            func = method_map[method_name]
            res = func(
                x["beta.exposure"].values,
                x["beta.outcome"].values,
                x["se.exposure"].values,
                x["se.outcome"].values,
                parameters
            )
            res_list.append(res)

        methl = mr_method_list()
        method_names = [
            next((m["name"] for m in methl if m["obj"] == meth), meth)
            for meth in method_list
        ]

        het_tab = pd.DataFrame({
            "id.exposure": [id_exp] * len(res_list),
            "id.outcome": [id_out] * len(res_list),
            "outcome": [x["outcome"].iloc[0]] * len(res_list),
            "exposure": [x["exposure"].iloc[0]] * len(res_list),
            "method": method_names,
            "Q": [r.get("Q", np.nan) for r in res_list],
            "Q_df": [r.get("Q_df", np.nan) for r in res_list],
            "Q_pval": [r.get("Q_pval", np.nan) for r in res_list]
        })

        het_tab = het_tab[~(het_tab["Q"].isna() & het_tab["Q_df"].isna() & het_tab["Q_pval"].isna())]
        results.append(het_tab)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["id.exposure", "id.outcome", "outcome", "exposure", "method", "Q", "Q_df", "Q_pval"])


def mr_pleiotropy_test(dat):
    results = []

    grouped = dat.groupby(["id.exposure", "id.outcome"])

    for (id_exp, id_out), group in grouped:
        x = group[group["mr_keep"]].copy()

        if len(x) < 2:
            print(f"Not enough SNPs available for pleiotropy analysis of '{id_exp}' on '{id_out}'")
            continue

        res = mr_egger_regression(
            x["beta.exposure"].values,
            x["beta.outcome"].values,
            x["se.exposure"].values,
            x["se.outcome"].values,
            default_parameters()
        )

        out = pd.DataFrame({
            "id.exposure": [id_exp],
            "id.outcome": [id_out],
            "outcome": [x["outcome"].iloc[0]],
            "exposure": [x["exposure"].iloc[0]],
            "egger_intercept": [res["b_i"]],
            "se": [res["se_i"]],
            "pval": [res["pval_i"]]
        })

        results.append(out)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=["id.exposure", "id.outcome", "outcome", "exposure", "egger_intercept", "se", "pval"])