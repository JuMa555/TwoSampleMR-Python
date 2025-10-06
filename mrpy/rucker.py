import pandas as pd
import numpy as np
from scipy.stats import chi2, norm, t
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from mr import default_parameters

def isq(y, s):
    k = len(y)
    w = 1 / s**2
    sum_w = np.sum(w)
    mu_hat = np.sum(y * w) / sum_w
    Q = np.sum(w * (y - mu_hat)**2)
    Isq = (Q - (k - 1)) / Q if Q != 0 else 0
    return max(0, Isq)


def PM(y, s, alpha=0.1):
    k = len(y)
    df = k - 1
    sig = norm.ppf(1 - alpha / 2)
    low = chi2.ppf(alpha / 2, df)
    up = chi2.ppf(1 - alpha / 2, df)
    med = chi2.ppf(0.5, df)
    mn = df
    mode = df - 1
    quant = [low, mode, mn, med, up]
    L = len(quant)

    tausq_list = []
    mu_list = []
    isq_list = []
    CI = np.empty((L, 2))

    v = 1 / s**2
    sum_v = np.sum(v)
    typS = np.sum(v * (k - 1)) / (sum_v**2 - np.sum(v**2))

    for q in quant:
        tausq = 0
        TAUsq = []
        Fstat = 1
        while Fstat > 0:
            TAUsq.append(tausq)
            w = 1 / (s**2 + tausq)
            sum_w = np.sum(w)
            w2 = w**2
            yW = np.sum(y * w) / sum_w
            Q1 = np.sum(w * (y - yW)**2)
            Q2 = np.sum(w2 * (y - yW)**2)
            Fstat = Q1 - q
            delta = Fstat / Q2 if Q2 != 0 else 0
            tausq += delta
        tausq = max(tausq, 0)
        tausq_list.append(tausq)
        mu_list.append(yW)
        V = 1 / sum_w
        isq_list.append(tausq / (tausq + typS) if (tausq + typS) != 0 else 0)
        CI_val = yW + sig * np.array([-1, 1]) * np.sqrt(V)
        CI[quant.index(q), :] = CI_val

    return {
        "tausq": tausq_list,
        "muhat": mu_list,
        "Isq": isq_list,
        "CI": CI,
        "quant": quant
    }

def mr_rucker(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    dat = dat[dat["mr_keep"]]

    unique_combos = dat.drop_duplicates(subset=["id.exposure", "id.outcome"])[
        ["exposure", "outcome", "id.exposure", "id.outcome"]
    ]

    results = []
    for _, row in unique_combos.iterrows():
        exposure = row["exposure"]
        outcome = row["outcome"]
        print(f"{exposure} - {outcome}")

        subset_data = dat[(dat["exposure"] == exposure) & (dat["outcome"] == outcome)]

        result = mr_rucker_internal(subset_data, parameters)
        result["id.exposure"] = row["id.exposure"]
        result["id.outcome"] = row["id.outcome"]
        result["exposure"] = exposure
        result["outcome"] = outcome

        results.append(result)

    return results


def mr_rucker_internal(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    if len(dat) < 3:
        print("Need at least 3 SNPs")
        return None

    def sign0(x):
        x = np.where(x == 0, 1, x)
        return np.sign(x)

    dat = dat.copy()
    dat["beta.outcome"] = dat["beta.outcome"] * sign0(dat["beta.exposure"])
    dat["beta.exposure"] = np.abs(dat["beta.exposure"])

    Qthresh = parameters.get("Qthresh", 0.05)
    alpha = parameters.get("alpha", 0.05)
    test_dist = parameters.get("test_dist", "z")

    nsnp = len(dat)
    b_exp = dat["beta.exposure"].values
    b_out = dat["beta.outcome"].values
    se_exp = dat["se.exposure"].values
    se_out = dat["se.outcome"].values

    w = b_exp**2 / se_out**2
    y = b_out / se_out
    x = b_exp / se_out
    i = 1 / se_out

    lmod_ivw = sm.OLS(y, x).fit()
    b_ivw_fe = lmod_ivw.params[0]
    se_ivw = lmod_ivw.bse[0]
    sigma_ivw = np.sqrt(lmod_ivw.mse_resid)
    Q_ivw = lmod_ivw.mse_resid * (nsnp - 1)
    Q_df_ivw = nsnp - 1
    Q_pval_ivw = chi2.sf(Q_ivw, Q_df_ivw)
    phi_ivw = Q_ivw / (nsnp - 1)

    se_ivw_fe = se_ivw / max(1, sigma_ivw)
    if test_dist == "z":
        pval_ivw_fe = 2 * norm.sf(abs(b_ivw_fe / se_ivw_fe))
    else:
        pval_ivw_fe = 2 * t.sf(abs(b_ivw_fe / se_ivw_fe), df=nsnp - 1)

    b_ivw_re = b_ivw_fe
    se_ivw_re = se_ivw
    if test_dist == "z":
        pval_ivw_re = 2 * norm.sf(abs(b_ivw_fe / se_ivw))
    else:
        pval_ivw_re = lmod_ivw.pvalues[0]

    X_egger = np.column_stack((i, x))
    lmod_egger = sm.OLS(y, X_egger).fit()
    sigma_egger = np.sqrt(lmod_egger.mse_resid)

    b0_egger_fe = lmod_egger.params[0]
    b1_egger_fe = lmod_egger.params[1]
    se0_egger_fe = lmod_egger.bse[0] / max(1, sigma_egger)
    se1_egger_fe = lmod_egger.bse[1] / max(1, sigma_egger)

    pval0_egger_fe = (
        2 * norm.sf(abs(b0_egger_fe / se0_egger_fe))
        if test_dist == "z"
        else 2 * t.sf(abs(b0_egger_fe / se0_egger_fe), df=nsnp - 2)
    )
    pval1_egger_fe = 2 * t.sf(abs(b1_egger_fe / se1_egger_fe), df=nsnp - 2)

    Q_egger = lmod_egger.mse_resid * (nsnp - 2)
    Q_df_egger = nsnp - 2
    Q_pval_egger = chi2.sf(Q_egger, Q_df_egger)
    phi_egger = Q_egger / Q_df_egger

    b0_egger_re = lmod_egger.params[0]
    b1_egger_re = lmod_egger.params[1]
    se0_egger_re = lmod_egger.bse[0]
    se1_egger_re = lmod_egger.bse[1]
    pval0_egger_re = (
        2 * norm.sf(abs(b0_egger_re / se0_egger_re))
        if test_dist == "z"
        else lmod_egger.pvalues[0]
    )
    pval1_egger_re = lmod_egger.pvalues[1]

    zscore = norm.ppf(1 - alpha / 2)

    results = pd.DataFrame({
        "Method": [
            "IVW fixed effects",
            "IVW random effects",
            "Egger fixed effects",
            "Egger random effects"
        ],
        "nsnp": nsnp,
        "Estimate": [b_ivw_fe, b_ivw_re, b1_egger_fe, b1_egger_re],
        "SE": [se_ivw_fe, se_ivw_re, se1_egger_fe, se1_egger_re]
    })
    results["CI_low"] = results["Estimate"] - zscore * results["SE"]
    results["CI_upp"] = results["Estimate"] + zscore * results["SE"]
    results["P"] = [pval_ivw_fe, pval_ivw_re, pval1_egger_fe, pval1_egger_re]

    Qdiff = max(0, Q_ivw - Q_egger)
    Qdiff_p = chi2.sf(Qdiff, df=1)

    Q = pd.DataFrame({
        "Method": ["Q_ivw", "Q_egger", "Q_diff"],
        "Q": [Q_ivw, Q_egger, Qdiff],
        "df": [Q_df_ivw, Q_df_egger, 1],
        "P": [Q_pval_ivw, Q_pval_egger, Qdiff_p]
    })

    intercept = pd.DataFrame({
        "Method": ["Egger fixed effects", "Egger random effects"],
        "Estimate": [b0_egger_fe, b0_egger_fe],
        "SE": [se0_egger_fe, se0_egger_re]
    })
    intercept["CI_low"] = intercept["Estimate"] - zscore * intercept["SE"]
    intercept["CI_upp"] = intercept["Estimate"] + zscore * intercept["SE"]
    intercept["P"] = [pval0_egger_fe, pval0_egger_re]

    if Q_pval_ivw <= Qthresh:
        if Qdiff_p <= Qthresh:
            res = "D" if Q_pval_egger <= Qthresh else "C"
        else:
            res = "B"
    else:
        res = "A"

    selected = results[results["Method"].str.contains({
        "A": "IVW fixed",
        "B": "IVW random",
        "C": "Egger fixed",
        "D": "Egger random"
    }[res])]
    selected = selected.copy()
    selected["Method"] = "Rucker"

    cooksdistance = (
        sm.OLS(y, x).fit().get_influence().cooks_distance[0]
        if res in ["A", "B"]
        else sm.OLS(y, X_egger).fit().get_influence().cooks_distance[0]
    )

    return {
        "rucker": results,
        "intercept": intercept,
        "Q": Q,
        "res": res,
        "selected": selected,
        "cooksdistance": cooksdistance,
        "lmod_ivw": sm.OLS(y, x).fit(),
        "lmod_egger": sm.OLS(y, X_egger).fit()
    }


def mr_rucker_bootstrap(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    nboot = parameters["nboot"]
    nsnp = len(dat)
    Qthresh = parameters["Qthresh"]

    rucker = mr_rucker_internal(dat, parameters)
    dat2 = dat.copy()
    l = []

    for _ in range(nboot):
        dat2["beta.exposure"] = np.random.normal(dat["beta.exposure"], dat["se.exposure"])
        dat2["beta.outcome"] = np.random.normal(dat["beta.outcome"], dat["se.outcome"])
        l.append(mr_rucker_internal(dat2, parameters))

    modsel = pd.concat([r["selected"].assign(model=r["res"]) for r in l], ignore_index=True)

    bootstrap = pd.DataFrame({
        "Q": [rucker["Q"]["Q"].iloc[0]] + [r["Q"]["Q"].iloc[0] for r in l],
        "Qdash": [rucker["Q"]["Q"].iloc[1]] + [r["Q"]["Q"].iloc[1] for r in l],
        "model": [rucker["res"]] + [r["res"] for r in l],
        "i": ["Full"] + ["Bootstrap"] * nboot
    })

    rucker_point = rucker["selected"].copy()
    rucker_point["Method"] = "Rucker point estimate"

    rucker_median = pd.DataFrame([{
        "Method": "Rucker median",
        "nsnp": nsnp,
        "Estimate": modsel["Estimate"].median(),
        "SE": modsel["Estimate"].mad(),
        "CI_low": modsel["Estimate"].quantile(0.025),
        "CI_upp": modsel["Estimate"].quantile(0.975)
    }])
    rucker_median["P"] = 2 * t.sf(np.abs(rucker_median["Estimate"] / rucker_median["SE"]), df=nsnp - 1)

    rucker_mean = pd.DataFrame([{
        "Method": "Rucker mean",
        "nsnp": nsnp,
        "Estimate": modsel["Estimate"].mean(),
        "SE": modsel["Estimate"].std()
    }])
    qz = norm.ppf(1 - Qthresh / 2)
    rucker_mean["CI_low"] = rucker_mean["Estimate"] - qz * rucker_mean["SE"]
    rucker_mean["CI_upp"] = rucker_mean["Estimate"] + qz * rucker_mean["SE"]
    rucker_mean["P"] = 2 * t.sf(np.abs(rucker_mean["Estimate"] / rucker_mean["SE"]), df=nsnp - 1)

    res = pd.concat([rucker["rucker"], rucker_point, rucker_mean, rucker_median], ignore_index=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=bootstrap, x="Q", y="Qdash", hue="model", style="i")
    qcut = chi2.ppf(1 - Qthresh, 1)
    qcut_ivw = chi2.ppf(1 - Qthresh, df=nsnp - 1)
    qcut_egger = chi2.ppf(1 - Qthresh, df=nsnp - 2)
    max_q = max(bootstrap[["Q", "Qdash"]].max())
    plt.plot([0, max_q], [0, max_q], color="grey")
    plt.axline((0, qcut), slope=1, linestyle="dotted")
    plt.axhline(y=qcut_egger, linestyle="dotted")
    plt.axvline(x=qcut_ivw, linestyle="dotted")
    plt.xlabel("Q")
    plt.ylabel("Q'")
    plt.title("Rucker Q plot")
    plt.legend()
    q_plot = plt.gcf()
    plt.close()


    modsel["model_name"] = "IVW"
    modsel.loc[modsel["model"].isin(["C", "D"]), "model_name"] = "Egger"

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=modsel, x="Estimate", hue="model_name", fill=True, alpha=0.4)
    for _, row in res.iterrows():
        plt.axvline(x=row["Estimate"], label=row["Method"])
    plt.xlabel("Estimate")
    plt.ylabel("Density")
    plt.title("Bootstrap estimate distribution")
    plt.legend()
    e_plot = plt.gcf()
    plt.close()

    return {
        "rucker": rucker,
        "res": res,
        "bootstrap_estimates": modsel,
        "boostrap_q": bootstrap,
        "q_plot": q_plot,
        "e_plot": e_plot
    }


def mr_rucker_jackknife(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    dat = dat[dat["mr_keep"]]

    unique_combos = dat.drop_duplicates(subset=["id.exposure", "id.outcome"])[
        ["exposure", "outcome", "id.exposure", "id.outcome"]
    ]

    results = []
    for _, row in unique_combos.iterrows():
        exposure = row["exposure"]
        outcome = row["outcome"]
        print(f"{exposure} - {outcome}")

        subset_data = dat[(dat["exposure"] == exposure) & (dat["outcome"] == outcome)]

        result = mr_rucker_jackknife_internal(subset_data, parameters)
        result["id.exposure"] = row["id.exposure"]
        result["id.outcome"] = row["id.outcome"]
        result["exposure"] = exposure
        result["outcome"] = outcome

        results.append(result)

    return results


def mr_rucker_jackknife_internal(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    nboot = parameters["nboot"]
    nsnp = len(dat)
    Qthresh = parameters["Qthresh"]

    rucker = mr_rucker_internal(dat, parameters)
    rucker_point = rucker["selected"].copy()
    rucker_point["Method"] = "Rucker point estimate"

    if nsnp < 15:
        print("Too few SNPs for jackknife")
        res = pd.concat([rucker["rucker"], rucker_point], ignore_index=True)
        return {
            "rucker": rucker,
            "res": res,
            "bootstrap_estimates": None,
            "boostrap_q": None,
            "q_plot": None,
            "e_plot": None
        }

    l = []
    for _ in range(nboot):
        dat2 = dat.sample(n=nsnp, replace=True)
        l.append(mr_rucker_internal(dat2, parameters))

    modsel = pd.concat([r["selected"].assign(model=r["res"]) for r in l], ignore_index=True)

    bootstrap = pd.DataFrame({
        "Q": [rucker["Q"]["Q"].iloc[0]] + [r["Q"]["Q"].iloc[0] for r in l],
        "Qdash": [rucker["Q"]["Q"].iloc[1]] + [r["Q"]["Q"].iloc[1] for r in l],
        "model": [rucker["res"]] + [r["res"] for r in l],
        "i": ["Full"] + ["Jackknife"] * nboot
    })

    rucker_median = pd.DataFrame([{
        "Method": "Rucker median (JK)",
        "nsnp": nsnp,
        "Estimate": modsel["Estimate"].median(),
        "SE": modsel["Estimate"].mad(),
        "CI_low": modsel["Estimate"].quantile(0.025),
        "CI_upp": modsel["Estimate"].quantile(0.975)
    }])
    rucker_median["P"] = 2 * t.sf(np.abs(rucker_median["Estimate"] / rucker_median["SE"]), df=nsnp - 1)

    rucker_mean = pd.DataFrame([{
        "Method": "Rucker mean (JK)",
        "nsnp": nsnp,
        "Estimate": modsel["Estimate"].mean(),
        "SE": modsel["Estimate"].std()
    }])
    qz = norm.ppf(1 - Qthresh / 2)
    rucker_mean["CI_low"] = rucker_mean["Estimate"] - qz * rucker_mean["SE"]
    rucker_mean["CI_upp"] = rucker_mean["Estimate"] + qz * rucker_mean["SE"]
    rucker_mean["P"] = 2 * t.sf(np.abs(rucker_mean["Estimate"] / rucker_mean["SE"]), df=nsnp - 1)

    res = pd.concat([rucker["rucker"], rucker_point, rucker_mean, rucker_median], ignore_index=True)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=bootstrap, x="Q", y="Qdash", hue="model", style="i")
    qcut = chi2.ppf(1 - Qthresh, 1)
    qcut_ivw = chi2.ppf(1 - Qthresh, df=nsnp - 1)
    qcut_egger = chi2.ppf(1 - Qthresh, df=nsnp - 2)
    max_q = max(bootstrap[["Q", "Qdash"]].max())
    plt.plot([0, max_q], [0, max_q], color="grey")
    plt.axline((0, qcut), slope=1, linestyle="dotted")
    plt.axhline(y=qcut_egger, linestyle="dotted")
    plt.axvline(x=qcut_ivw, linestyle="dotted")
    plt.xlabel("Q")
    plt.ylabel("Q'")
    plt.title("Rucker Q plot (Jackknife)")
    plt.legend()
    q_plot = plt.gcf()
    plt.close()

    modsel["model_name"] = "IVW"
    modsel.loc[modsel["model"].isin(["C", "D"]), "model_name"] = "Egger"

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=modsel, x="Estimate", hue="model_name", fill=True, alpha=0.4)
    for _, row in res.iterrows():
        plt.axvline(x=row["Estimate"], label=row["Method"])
    plt.xlabel("Estimate")
    plt.ylabel("Density")
    plt.title("Jackknife estimate distribution")
    plt.legend()
    e_plot = plt.gcf()
    plt.close()

    return {
        "rucker": rucker,
        "res": res,
        "bootstrap_estimates": modsel,
        "boostrap_q": bootstrap,
        "q_plot": q_plot,
        "e_plot": e_plot
    }


def mr_rucker_cooksdistance(dat, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    dat_orig = dat.copy()
    rucker_orig = mr_rucker_internal(dat_orig, parameters)
    rucker = rucker_orig

    cooks_threshold = 4 / len(dat)
    index = rucker_orig["cooksdistance"] > cooks_threshold

    i = 1
    history = []

    while any(index) and (len(dat[~index]) > 3):
        dat = dat[~index]
        cooks_threshold = 4 / len(dat)
        rucker = mr_rucker_internal(dat, parameters)
        history.append(rucker)
        index = rucker["cooksdistance"] > cooks_threshold
        i += 1

    removed_snps = dat_orig.loc[~dat_orig["SNP"].isin(dat["SNP"]), "SNP"].tolist()

    rucker["removed_snps"] = removed_snps
    rucker["selected"] = rucker["selected"].copy()
    rucker["selected"]["Method"] = "Rucker (CD)"

    rucker["rucker"] = rucker["rucker"].copy()
    rucker["rucker"]["Method"] = rucker["rucker"]["Method"].astype(str) + " (CD)"

    return rucker