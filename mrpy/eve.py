import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from mr import mr_wald_ratio, default_parameters, mr_median
from mr_mode import mr_mode
from moe import system_metrics
from steiger_filtering import steiger_filtering

def mr_mean_ivw(dat):
    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    if len(dat) < 1:
        raise ValueError("No SNPs retained for MR analysis.")

    b_exp = dat["beta.exposure"].values
    b_out = dat["beta.outcome"].values
    se_exp = dat["se.exposure"].values
    se_out = dat["se.outcome"].values
    ratios = b_out / b_exp

    if len(set(dat["id.exposure"])) > 1 or len(set(dat["id.outcome"])) > 1:
        raise ValueError("Only one exposure and one outcome allowed.")

    id_exposure = dat["id.exposure"].iloc[0]
    id_outcome = dat["id.outcome"].iloc[0]

    if len(dat) == 1:
        res = mr_wald_ratio(b_exp, b_out, se_exp, se_out, parameters=default_parameters())
        b, se, pval = res["b"], res["se"], res["pval"]
        ci_low = b - 1.96 * se
        ci_upp = b + 1.96 * se
        estimates = pd.DataFrame([{
            "id.exposure": id_exposure,
            "id.outcome": id_outcome,
            "method": "Wald ratio",
            "nsnp": 1,
            "b": b,
            "se": se,
            "ci_low": ci_low,
            "ci_upp": ci_upp,
            "pval": pval
        }])
        return {"estimates": estimates}

    X_unw = b_exp.reshape(-1, 1)
    model_unw = sm.OLS(b_out, X_unw).fit()
    b_unw = model_unw.params[0]
    se_unw = model_unw.bse[0]
    ci_low_unw = b_unw - 1.96 * se_unw
    ci_upp_unw = b_unw + 1.96 * se_unw
    pval_unw = 2 * stats.t.sf(abs(b_unw / se_unw), df=len(b_exp)-1)
    unw_out = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "Simple mean",
        "nsnp": len(dat),
        "b": b_unw,
        "se": se_unw,
        "ci_low": ci_low_unw,
        "ci_upp": ci_upp_unw,
        "pval": pval_unw
    }])

    weights1 = np.sqrt(b_exp**2 / se_out**2)
    y1 = ratios * weights1
    X1 = weights1.reshape(-1, 1)
    model_ivw1 = sm.OLS(y1, X1).fit()
    b_ivw1 = model_ivw1.params[0]

    weights2 = 1 / ((se_out**2 + (b_ivw1**2) * se_exp**2) / b_exp**2)
    y2 = ratios * np.sqrt(weights2)
    X2 = np.sqrt(weights2).reshape(-1, 1)
    model_ivw2 = sm.OLS(y2, X2).fit()
    b_ivw2 = model_ivw2.params[0]
    se_ivw2 = model_ivw2.bse[0]

    Qj = weights2 * (ratios - b_ivw2)**2
    Qivw2 = np.sum(Qj)
    Qivw2pval = stats.chi2.sf(Qivw2, df=len(dat)-1)
    outliers = pd.DataFrame({
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "SNP": dat["SNP"],
        "Qj": Qj,
        "Qpval": stats.chi2.sf(Qj, df=1)
    })

    ci_low = b_ivw2 - 1.96 * se_ivw2
    ci_upp = b_ivw2 + 1.96 * se_ivw2
    pval = 2 * stats.t.sf(abs(b_ivw2 / se_ivw2), df=len(dat)-1)
    re_out = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "RE IVW",
        "nsnp": len(dat),
        "b": b_ivw2,
        "se": se_ivw2,
        "ci_low": ci_low,
        "ci_upp": ci_upp,
        "pval": pval
    }])

    fe_se = se_ivw2 / max(1, model_ivw2.mse_resid**0.5)
    fe_pval = 2 * norm.sf(abs(b_ivw2 / fe_se))
    fe_out = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "FE IVW",
        "nsnp": len(dat),
        "b": b_ivw2,
        "se": fe_se,
        "ci_low": b_ivw2 - 1.96 * fe_se,
        "ci_upp": b_ivw2 + 1.96 * fe_se,
        "pval": fe_pval
    }])

    heterogeneity = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "IVW",
        "Q": Qivw2,
        "df": len(dat) - 1,
        "pval": Qivw2pval
    }])

    estimates = pd.concat([unw_out, fe_out, re_out], ignore_index=True)

    return {"estimates": estimates, "heterogeneity": heterogeneity, "outliers": outliers}


def mr_mean_egger(dat):
    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    if len(dat) < 3:
        raise ValueError("Need at least 3 SNPs for Egger regression.")

    b_exp = dat["beta.exposure"].values
    b_out = dat["beta.outcome"].values
    se_exp = dat["se.exposure"].values
    se_out = dat["se.outcome"].values
    ratios = b_out / b_exp

    if len(set(dat["id.exposure"])) > 1 or len(set(dat["id.outcome"])) > 1:
        raise ValueError("Only one exposure and one outcome allowed.")

    id_exposure = dat["id.exposure"].iloc[0]
    id_outcome = dat["id.outcome"].iloc[0]

    weights1 = np.sqrt(b_exp**2 / se_out**2)
    y1 = ratios * weights1
    X1 = sm.add_constant(weights1)
    model_egger1 = sm.OLS(y1, X1).fit()
    intercept_1, slope_1 = model_egger1.params

    weights2 = 1 / ((se_out**2 + slope_1**2 * se_exp**2) / b_exp**2)
    weights2_sqrt = np.sqrt(weights2)
    y2 = ratios * weights2_sqrt
    X2 = sm.add_constant(weights2_sqrt)
    model_egger2 = sm.OLS(y2, X2).fit()
    intercept_2, slope_2 = model_egger2.params
    se_intercept_2, se_slope_2 = model_egger2.bse
    sigma = np.sqrt(model_egger2.mse_resid)

    Qj = weights2 * (ratios - intercept_2 / weights2_sqrt - slope_2)**2
    Qpval = stats.chi2.sf(Qj, df=1)
    outliers = pd.DataFrame({
        "SNP": dat["SNP"],
        "Qj": Qj,
        "Qpval": Qpval,
        "id.exposure": id_exposure,
        "id.outcome": id_outcome
    })

    Q = np.sum(Qj)
    Q_pval = stats.chi2.sf(Q, df=len(dat)-2)

    ci_low = slope_2 - 1.96 * se_slope_2
    ci_upp = slope_2 + 1.96 * se_slope_2
    pval_slope = 2 * stats.t.sf(abs(slope_2 / se_slope_2), df=len(dat) - 2)
    re_out = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "RE Egger",
        "nsnp": len(dat),
        "b": slope_2,
        "se": se_slope_2,
        "ci_low": ci_low,
        "ci_upp": ci_upp,
        "pval": pval_slope
    }])

    fe_se = se_slope_2 / max(1, sigma)
    fe_pval = 2 * stats.norm.sf(abs(slope_2 / fe_se))
    fe_out = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "FE Egger",
        "nsnp": len(dat),
        "b": slope_2,
        "se": fe_se,
        "ci_low": slope_2 - 1.96 * fe_se,
        "ci_upp": slope_2 + 1.96 * fe_se,
        "pval": fe_pval
    }])

    estimates = pd.concat([fe_out, re_out], ignore_index=True)

    heterogeneity = pd.DataFrame([{
        "id.exposure": id_exposure,
        "id.outcome": id_outcome,
        "method": "Egger",
        "Q": Q,
        "df": len(dat) - 2,
        "pval": Q_pval
    }])

    pleio_pval_fe = 2 * stats.t.sf(abs(intercept_2 / (se_intercept_2 / max(1, sigma))), df=len(dat) - 2)
    pleio_pval_re = 2 * stats.t.sf(abs(intercept_2 / se_intercept_2), df=len(dat) - 2)

    directional_pleiotropy = pd.DataFrame({
        "id.exposure": [id_exposure] * 2,
        "id.outcome": [id_outcome] * 2,
        "method": ["FE Egger intercept", "RE Egger intercept"],
        "nsnp": [len(dat)] * 2,
        "b": [intercept_2, intercept_2],
        "se": [se_intercept_2 / max(1, sigma), se_intercept_2],
        "pval": [pleio_pval_fe, pleio_pval_re]
    })

    return {
        "estimates": estimates,
        "heterogeneity": heterogeneity,
        "directional_pleiotropy": directional_pleiotropy,
        "outliers": outliers
    }

def mr_mean(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    try:
        m1 = mr_mean_ivw(dat)
    except Exception:
        return None

    try:
        m2 = mr_mean_egger(dat)
    except Exception:
        return m1

    estimates = pd.concat([m1["estimates"], m2["estimates"]], ignore_index=True)
    heterogeneity = pd.concat([m1["heterogeneity"], m2["heterogeneity"]], ignore_index=True)
    directional_pleiotropy = m2.get("directional_pleiotropy", None)
    outliers = m1.get("outliers", None)

    Q_diff = heterogeneity["Q"].iloc[0] - heterogeneity["Q"].iloc[1]
    pval = stats.chi2.sf(Q_diff, df=1)

    rucker = pd.DataFrame([{
        "id.exposure": dat["id.exposure"].iloc[0],
        "id.outcome": dat["id.outcome"].iloc[0],
        "method": "Rucker",
        "Q": Q_diff,
        "df": 1,
        "pval": pval
    }])

    heterogeneity = pd.concat([heterogeneity, rucker], ignore_index=True)

    return {
        "estimates": estimates,
        "heterogeneity": heterogeneity,
        "directional_pleiotropy": directional_pleiotropy,
        "outliers": outliers
    }

def mr_all(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    res = mr_mean(dat)

    if dat["mr_keep"].sum() > 3:
        try:
            median_res = mr_median(dat, parameters=parameters)
        except Exception:
            median_res = None

        try:
            mode_res = mr_mode(dat, parameters=parameters)
            mode_res = mode_res.iloc[:3]
        except Exception:
            mode_res = None

        estimates = [res["estimates"]]
        if median_res is not None:
            estimates.append(median_res)
        if mode_res is not None:
            estimates.append(mode_res)

        res["estimates"] = pd.concat(estimates, ignore_index=True)

    system_info = system_metrics(dat)

    Fstat = (dat["beta.exposure"]**2 / dat["se.exposure"]**2).replace([np.inf, -np.inf], 300)

    info = {
        "id.exposure": dat["id.exposure"].iloc[0],
        "id.outcome": dat["id.outcome"].iloc[0],
        "nsnp": len(dat),
        "nout": dat["samplesize.outcome"].mean(skipna=True),
        "nexp": dat["samplesize.exposure"].mean(skipna=True),
        "meanF": Fstat.mean(skipna=True),
        "varF": Fstat.var(skipna=True),
        "medianF": Fstat.median(skipna=True),
        **system_info,
        "steiger_filtered": dat.get("steiger_dir", pd.Series([False])).any(),
        "outlier_filtered": dat.get("outlier", pd.Series([False])).any(),
        "nsnp_removed": (~dat["mr_keep"]).sum()
    }

    ordered_cols = [
        "id.exposure", "id.outcome", "nsnp", "nout", "nexp", "meanF", "varF", "medianF",
        "egger_isq", "sct", "Isq", "Isqe", "Qdiff", "intercept",
        "dfb1_ivw", "dfb2_ivw", "dfb3_ivw", "cooks_ivw",
        "dfb1_egger", "dfb2_egger", "dfb3_egger", "cooks_egger",
        "homosc_ivw", "homosc_egg", "shap_ivw", "shap_egger", "ks_ivw", "ks_egger",
        "steiger_filtered", "outlier_filtered", "nsnp_removed"
    ]

    info_df = pd.DataFrame([info])
    info_df = info_df.reindex(columns=ordered_cols)

    res["info"] = info_df
    return res


def mr_wrapper_single(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    dat = steiger_filtering(dat)
    m = [None] * 4

    snps_retained = pd.DataFrame({
        "SNP": dat["SNP"],
        "outlier": False,
        "steiger": False,
        "both": False
    })

    m[0] = mr_all(dat, parameters=parameters)

    if m[0] is not None:

        if "outliers" in m[0]:
            outlier_snps = m[0]["outliers"].query("Qpval < 0.05")["SNP"]
            temp = dat[~dat["SNP"].isin(outlier_snps)]
            m[1] = mr_all(temp, parameters=parameters)
            snps_retained.loc[snps_retained["SNP"].isin(temp["SNP"]), "outlier"] = True
        else:
            m[1] = m[0]
            snps_retained["outlier"] = True

        dat_st = dat[dat["steiger_dir"]]
        snps_retained.loc[snps_retained["SNP"].isin(dat_st["SNP"]), "steiger"] = True

        if dat_st.empty:
            m[2] = m[3] = {
                "estimates": pd.DataFrame([{
                    "method": "Steiger null", "nsnp": 0, "b": 0,
                    "se": np.nan, "ci_low": np.nan, "ci_upp": np.nan, "pval": 1
                }])
            }
        else:
            m[2] = mr_all(dat_st, parameters=parameters)
            if "outliers" in m[2]:
                outlier_snps_st = m[2]["outliers"].query("Qpval < 0.05")["SNP"]
                temp = dat_st[~dat_st["SNP"].isin(outlier_snps_st)]
                m[3] = mr_all(temp, parameters=parameters)
                snps_retained.loc[snps_retained["SNP"].isin(temp["SNP"]), "both"] = True
            else:
                m[3] = m[2]
                snps_retained["both"] = True

    for i, steiger_filt, outlier_filt in [
        (0, False, False),
        (1, False, True),
        (2, True, False),
        (3, True, True)
    ]:
        if m[i] is not None:
            for k in m[i]:
                m[i][k]["steiger_filtered"] = steiger_filt
                m[i][k]["outlier_filtered"] = outlier_filt
                m[i][k]["id.exposure"] = dat["id.exposure"].iloc[0]
                m[i][k]["id.outcome"] = dat["id.outcome"].iloc[0]

    keys = set().union(*[mi.keys() for mi in m if mi is not None]) - {"outliers"}
    out = {}

    for key in keys:
        combined = pd.concat([mi[key] for mi in m if mi is not None and key in mi], ignore_index=True)
        out[key] = combined

    if "info" in out and "nsnp" in out["info"]:
        out["info"]["nsnp_removed"] = out["info"]["nsnp"].iloc[0] - out["info"]["nsnp"]

    out["snps_retained"] = snps_retained

    return out


def mr_wrapper(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    results = {}

    grouped = dat.groupby(["id.exposure", "id.outcome"])
    for (id_exposure, id_outcome), group in grouped:
        print(f"Performing MR analysis of '{id_exposure}' on '{id_outcome}'")
        d = group[group["mr_keep"]]
        o = mr_wrapper_single(d, parameters=parameters)
        results[(id_exposure, id_outcome)] = o

    return results