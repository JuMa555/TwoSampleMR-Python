import numpy as np
import pandas as pd
import ieugwaspy as igp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from scipy.stats import norm

from instruments import extract_instruments
from harmonise import harmonise_data
from read_data import convert_outcome_to_exposure
from query import extract_outcome_data
from ld import clump_data

def mv_extract_exposures(
    id_exposure,
    clump_r2=0.001,
    clump_kb=10000,
    harmonise_strictness=2,
    opengwas_jwt=None,
    find_proxies=True,
    force_server=False,
    pval_threshold=5e-8,
    pop="EUR",
    plink_bin=None,
    bfile=None
):
    if len(id_exposure) <= 1:
        raise ValueError("More than one exposure ID is required.")
    
    id_exposure = igp.backwards.legacy_ids(id_exposure)

    exposure_dat = extract_instruments(
        id_exposure,
        p1=pval_threshold,
        r2=clump_r2,
        kb=clump_kb,
        opengwas_jwt=opengwas_jwt,
        force_server=force_server
    )

    temp = exposure_dat.copy()
    temp["id.exposure"] = 1
    temp = temp.sort_values(by="pval.exposure")
    temp = temp[~temp["SNP"].duplicated()]

    temp = clump_data(
        temp,
        clump_p1=pval_threshold,
        clump_r2=clump_r2,
        clump_kb=clump_kb,
        pop=pop,
        plink_bin=plink_bin,
        bfile=bfile
    )

    exposure_dat = exposure_dat[exposure_dat["SNP"].isin(temp["SNP"])]

    d1 = extract_outcome_data(
        exposure_dat["SNP"].tolist(),
        id_exposure,
        proxies=True
    )

    if len(d1["id.outcome"].unique()) != len(set(id_exposure)):
        raise ValueError("Mismatch in outcome IDs.")

    d1 = d1[d1["mr_keep.outcome"]]
    d2 = d1[d1["id.outcome"] != id_exposure[0]]
    d1 = convert_outcome_to_exposure(d1[d1["id.outcome"] == id_exposure[0]])

    d = harmonise_data(d1, d2, action=harmonise_strictness)
    d = d[d["mr_keep"]]

    snp_counts = d["SNP"].value_counts()
    keepsnps = snp_counts[snp_counts == len(id_exposure) - 1].index
    d = d[d["SNP"].isin(keepsnps)]

    dh1 = d[d["id.outcome"] == d["id.outcome"].iloc[0]][[
        "SNP", "exposure", "id.exposure", "effect_allele.exposure", "other_allele.exposure",
        "eaf.exposure", "beta.exposure", "se.exposure", "pval.exposure"
    ]]
    dh2 = d[[
        "SNP", "outcome", "id.outcome", "effect_allele.outcome", "other_allele.outcome",
        "eaf.outcome", "beta.outcome", "se.outcome", "pval.outcome"
    ]].rename(columns=lambda x: x.replace("outcome", "exposure"))

    dh = pd.concat([dh1, dh2], ignore_index=True)
    dh = dh.sort_values(by=["SNP", "exposure"], ascending=True)
    return dh


def mv_harmonise_data(exposure_dat: pd.DataFrame, outcome_dat: pd.DataFrame, harmonise_strictness=2):
    required_cols = [
        "SNP", "id.exposure", "exposure",
        "effect_allele.exposure", "beta.exposure",
        "se.exposure", "pval.exposure"
    ]
    if not all(col in exposure_dat.columns for col in required_cols):
        raise ValueError("Exposure dataset does not contain required columns.")

    nexposures = exposure_dat["id.exposure"].nunique()
    if nexposures <= 1:
        raise ValueError("More than one exposure required.")

    snp_counts = exposure_dat["SNP"].value_counts()
    keepsnps = snp_counts[snp_counts == nexposures].index
    exposure_dat = exposure_dat[exposure_dat["SNP"].isin(keepsnps)]

    first_exposure_id = exposure_dat["id.exposure"].iloc[0]
    harmonised = harmonise_data(
        exposure_dat[exposure_dat["id.exposure"] == first_exposure_id],
        outcome_dat,
        action=harmonise_strictness
    )

    harmonised = harmonised[harmonised["mr_keep"]]
    harmonised["SNP"] = harmonised["SNP"].astype(str)

    def wide_format(df, value_col):
        wide = df.pivot(index="SNP", columns="id.exposure", values=value_col)
        wide = wide.loc[wide.index.isin(harmonised["SNP"])]
        wide.index = wide.index.astype(str)
        return wide

    exposure_beta = wide_format(exposure_dat, "beta.exposure")
    exposure_pval = wide_format(exposure_dat, "pval.exposure")
    exposure_se = wide_format(exposure_dat, "se.exposure")

    harmonised = harmonised.set_index("SNP").loc[exposure_beta.index].reset_index()

    if not all(harmonised["SNP"] == exposure_beta.index):
        raise ValueError("SNP mismatch after harmonisation")

    outcome_beta = harmonised["beta.outcome"].values
    outcome_se = harmonised["se.outcome"].values
    outcome_pval = harmonised["pval.outcome"].values

    expname = exposure_dat[["id.exposure", "exposure"]].drop_duplicates()
    outname = outcome_dat[["id.outcome", "outcome"]].drop_duplicates()

    return {
        "exposure_beta": exposure_beta.to_numpy(),
        "exposure_pval": exposure_pval.to_numpy(),
        "exposure_se": exposure_se.to_numpy(),
        "outcome_beta": outcome_beta,
        "outcome_pval": outcome_pval,
        "outcome_se": outcome_se,
        "expname": expname,
        "outname": outname
    }


def mv_residual(mvdat, intercept=False, instrument_specific=False, pval_threshold=5e-8, plots=False):
    beta_outcome = mvdat["outcome_beta"]
    beta_exposure = mvdat["exposure_beta"]
    pval_exposure = mvdat["exposure_pval"]

    nexp = beta_exposure.shape[1]
    effs = np.full(nexp, np.nan)
    se = np.full(nexp, np.nan)
    pval = np.full(nexp, np.nan)
    nsnp = np.zeros(nexp, dtype=int)
    marginal_outcome = np.zeros_like(beta_exposure)
    plot_list = []

    nom = mvdat["expname"]["id.exposure"].values
    nom2_map = dict(zip(mvdat["expname"]["id.exposure"], mvdat["expname"]["exposure"]))
    nom2 = [nom2_map[n] for n in nom]

    for i in range(nexp):
        index = pval_exposure[:, i] < pval_threshold

        X_all = beta_exposure[:, np.arange(nexp) != i]
        y = beta_outcome

        if instrument_specific:
            X_sel = X_all[index]
            y_sel = y[index]

            if intercept:
                model1 = sm.OLS(y_sel, sm.add_constant(X_sel)).fit()
                residuals = model1.resid
                model2 = sm.OLS(residuals, sm.add_constant(beta_exposure[index, i])).fit()
            else:
                model1 = sm.OLS(y_sel, X_sel).fit()
                residuals = model1.resid
                model2 = sm.OLS(residuals, beta_exposure[index, i]).fit()

            marginal_outcome[index, i] = residuals
        else:
            if intercept:
                model1 = sm.OLS(y, sm.add_constant(X_all)).fit()
                residuals = model1.resid
                model2 = sm.OLS(residuals, sm.add_constant(beta_exposure[:, i])).fit()
            else:
                model1 = sm.OLS(y, X_all).fit()
                residuals = model1.resid
                model2 = sm.OLS(residuals, beta_exposure[:, i]).fit()

            marginal_outcome[:, i] = residuals

        if np.sum(index) > (nexp + int(intercept)):
            coeff_index = int(intercept)
            effs[i] = model2.params[coeff_index]
            se[i] = model2.bse[coeff_index]
        else:
            effs[i] = np.nan
            se[i] = np.nan

        if not np.isnan(effs[i]) and not np.isnan(se[i]):
            pval[i] = 2 * norm.sf(abs(effs[i] / se[i]))
        else:
            pval[i] = np.nan

        nsnp[i] = np.sum(index)

        if plots:
            d = pd.DataFrame({
                "outcome": marginal_outcome[:, i],
                "exposure": beta_exposure[:, i]
            })
            flip = np.sign(d["exposure"]) == -1
            d.loc[flip, "outcome"] *= -1
            d["exposure"] = np.abs(d["exposure"])

            fig, ax = plt.subplots()
            ax.scatter(d.loc[index, "exposure"], d.loc[index, "outcome"])
            ax.axline((0, 0), slope=effs[i])
            ax.set_xlabel(f"SNP effect on {nom2[i]}")
            ax.set_ylabel("Marginal SNP effect on outcome")
            plot_list.append(fig)

    result = pd.DataFrame({
        "id.exposure": nom,
        "id.outcome": mvdat["outname"]["id.outcome"].iloc[0],
        "outcome": mvdat["outname"]["outcome"].iloc[0],
        "nsnp": nsnp,
        "b": effs,
        "se": se,
        "pval": pval
    })

    result = mvdat["expname"].merge(result, on="id.exposure")
    out = {"result": result, "marginal_outcome": marginal_outcome}
    if plots:
        out["plots"] = plot_list

    return out


def mv_multiple(mvdat, intercept=False, instrument_specific=False, pval_threshold=5e-8, plots=False):
    beta_outcome = mvdat["outcome_beta"]
    beta_exposure = mvdat["exposure_beta"]
    pval_exposure = mvdat["exposure_pval"]
    w = 1 / mvdat["outcome_se"]**2

    nexp = beta_exposure.shape[1]
    effs = np.full(nexp, np.nan)
    se = np.full(nexp, np.nan)
    pval = np.full(nexp, np.nan)
    nsnp = np.zeros(nexp, dtype=int)
    plot_list = []

    nom = mvdat["expname"]["id.exposure"].values
    nom2_map = dict(zip(mvdat["expname"]["id.exposure"], mvdat["expname"]["exposure"]))
    nom2 = [nom2_map[n] for n in nom]

    for i in range(nexp):
        index = pval_exposure[:, i] < pval_threshold

        X = beta_exposure[index] if instrument_specific else beta_exposure
        y = beta_outcome[index] if instrument_specific else beta_outcome
        weights = w[index] if instrument_specific else w

        if not intercept:
            model = sm.WLS(y, X, weights=weights).fit()
        else:
            model = sm.WLS(y, sm.add_constant(X), weights=weights).fit()

        if instrument_specific and np.sum(index) <= (nexp + int(intercept)):
            effs[i] = np.nan
            se[i] = np.nan
        else:
            coef_idx = i + int(intercept)
            effs[i] = model.params[coef_idx]
            se[i] = model.bse[coef_idx]

        if not np.isnan(effs[i]) and not np.isnan(se[i]):
            pval[i] = 2 * norm.sf(abs(effs[i] / se[i]))
        else:
            pval[i] = np.nan

        nsnp[i] = np.sum(index)

        if plots:
            d = pd.DataFrame({
                "outcome": beta_outcome,
                "exposure": beta_exposure[:, i]
            })
            flip = np.sign(d["exposure"]) == -1
            d.loc[flip, "outcome"] *= -1
            d["exposure"] = np.abs(d["exposure"])

            fig, ax = plt.subplots()
            ax.scatter(d.loc[index, "exposure"], d.loc[index, "outcome"])
            ax.axline((0, 0), slope=effs[i])
            ax.set_xlabel(f"SNP effect on {nom2[i]}")
            ax.set_ylabel("Marginal SNP effect on outcome")
            plot_list.append(fig)

    result = pd.DataFrame({
        "id.exposure": nom,
        "id.outcome": mvdat["outname"]["id.outcome"].iloc[0],
        "outcome": mvdat["outname"]["outcome"].iloc[0],
        "nsnp": nsnp,
        "b": effs,
        "se": se,
        "pval": pval
    })

    result = mvdat["expname"].merge(result, on="id.exposure")
    out = {"result": result}
    if plots:
        out["plots"] = plot_list

    return out


def mv_basic(mvdat, pval_threshold=5e-8, plots=True):
    beta_outcome = mvdat["outcome_beta"]
    beta_exposure = mvdat["exposure_beta"]
    pval_exposure = mvdat["exposure_pval"]

    nexp = beta_exposure.shape[1]
    effs = np.full(nexp, np.nan)
    se = np.full(nexp, np.nan)
    pval = np.full(nexp, np.nan)
    nsnp = np.zeros(nexp, dtype=int)
    marginal_outcome = np.zeros_like(beta_exposure)
    plot_list = []

    nom = mvdat["expname"]["id.exposure"].values
    nom2_map = dict(zip(mvdat["expname"]["id.exposure"], mvdat["expname"]["exposure"]))
    nom2 = [nom2_map[n] for n in nom]

    for i in range(nexp):
        index = pval_exposure[:, i] < pval_threshold

        X_other = np.delete(beta_exposure, i, axis=1)
        model_resid = sm.OLS(beta_outcome, sm.add_constant(X_other)).fit()
        marginal_outcome[:, i] = model_resid.resid

        model = sm.OLS(marginal_outcome[index, i], sm.add_constant(beta_exposure[index, i])).fit()
        
        effs[i] = model.params[1]
        se[i] = model.bse[1]
        pval[i] = 2 * norm.sf(abs(effs[i] / se[i]))
        nsnp[i] = np.sum(index)

        if plots:
            d = pd.DataFrame({
                "outcome": marginal_outcome[:, i],
                "exposure": beta_exposure[:, i]
            })
            flip = np.sign(d["exposure"]) == -1
            d.loc[flip, "outcome"] *= -1
            d["exposure"] = np.abs(d["exposure"])

            fig, ax = plt.subplots()
            ax.scatter(d.loc[index, "exposure"], d.loc[index, "outcome"])
            ax.axline((0, 0), slope=effs[i])
            ax.set_xlabel(f"SNP effect on {nom2[i]}")
            ax.set_ylabel("Marginal SNP effect on outcome")
            plot_list.append(fig)

    result = pd.DataFrame({
        "id.exposure": nom,
        "id.outcome": mvdat["outname"]["id.outcome"].iloc[0],
        "outcome": mvdat["outname"]["outcome"].iloc[0],
        "nsnp": nsnp,
        "b": effs,
        "se": se,
        "pval": pval
    })

    result = mvdat["expname"].merge(result, on="id.exposure")

    out = {"result": result, "marginal_outcome": marginal_outcome}
    if plots:
        out["plots"] = plot_list

    return out


def mv_ivw(mvdat, pval_threshold=5e-8, plots=True):
    beta_outcome = mvdat["outcome_beta"]
    beta_exposure = mvdat["exposure_beta"]
    pval_exposure = mvdat["exposure_pval"]
    w = 1 / mvdat["outcome_se"]**2

    nexp = beta_exposure.shape[1]
    effs = np.full(nexp, np.nan)
    se = np.full(nexp, np.nan)
    pval = np.full(nexp, np.nan)
    nsnp = np.zeros(nexp, dtype=int)
    plot_list = []

    nom = mvdat["expname"]["id.exposure"].values
    nom2_map = dict(zip(mvdat["expname"]["id.exposure"], mvdat["expname"]["exposure"]))
    nom2 = [nom2_map[n] for n in nom]

    for i in range(nexp):
        index = pval_exposure[:, i] < pval_threshold

        X = beta_exposure[index]
        y = beta_outcome[index]
        weights = w[index]

        model = sm.WLS(y, X, weights=weights).fit()

        effs[i] = model.params[i]
        se[i] = model.bse[i]
        pval[i] = 2 * norm.sf(abs(effs[i] / se[i]))
        nsnp[i] = np.sum(index)

        if plots:
            d = pd.DataFrame({
                "outcome": beta_outcome,
                "exposure": beta_exposure[:, i]
            })
            flip = np.sign(d["exposure"]) == -1
            d.loc[flip, "outcome"] *= -1
            d["exposure"] = np.abs(d["exposure"])

            fig, ax = plt.subplots()
            ax.scatter(d.loc[index, "exposure"], d.loc[index, "outcome"])
            ax.axline((0, 0), slope=effs[i])
            ax.set_xlabel(f"SNP effect on {nom2[i]}")
            ax.set_ylabel("Marginal SNP effect on outcome")
            plot_list.append(fig)

    result = pd.DataFrame({
        "id.exposure": nom,
        "id.outcome": mvdat["outname"]["id.outcome"].iloc[0],
        "outcome": mvdat["outname"]["outcome"].iloc[0],
        "nsnp": nsnp,
        "b": effs,
        "se": se,
        "pval": pval
    })

    result = mvdat["expname"].merge(result, on="id.exposure")

    out = {"result": result}
    if plots:
        out["plots"] = plot_list

    return out

def mv_lasso_feature_selection(mvdat):
    print("Performing feature selection")
    
    X = mvdat["exposure_beta"]
    y = mvdat["outcome_beta"]
    sample_weight = 1 / mvdat["outcome_se"]**2

    model = LassoCV(cv=10, fit_intercept=False).fit(X, y, sample_weight=sample_weight)

    coefs = model.coef_
    selected = coefs != 0
    selected_names = mvdat["expname"]["id.exposure"].values[selected]

    d = pd.DataFrame({
        "exposure": selected_names,
        "b": coefs[selected]
    })

    return d


def mv_subset(mvdat, features=None, intercept=False, instrument_specific=False, pval_threshold=5e-8, plots=False):
    if features is None:
        features = mv_lasso_feature_selection(mvdat)

    selected_exposures = features["exposure"].values

    mvdat["exposure_beta"] = mvdat["exposure_beta"][:, [i for i, col in enumerate(mvdat["expname"]["id.exposure"].values) if col in selected_exposures]]
    mvdat["exposure_se"] = mvdat["exposure_se"][:, [i for i, col in enumerate(mvdat["expname"]["id.exposure"].values) if col in selected_exposures]]
    mvdat["exposure_pval"] = mvdat["exposure_pval"][:, [i for i, col in enumerate(mvdat["expname"]["id.exposure"].values) if col in selected_exposures]]

    mvdat["expname"] = mvdat["expname"][mvdat["expname"]["id.exposure"].isin(selected_exposures)].reset_index(drop=True)

    instruments = np.any(mvdat["exposure_pval"] < pval_threshold, axis=1)
    if np.sum(instruments) <= len(selected_exposures):
        raise ValueError("Not enough instruments selected")

    mvdat["exposure_beta"] = mvdat["exposure_beta"][instruments]
    mvdat["exposure_se"] = mvdat["exposure_se"][instruments]
    mvdat["exposure_pval"] = mvdat["exposure_pval"][instruments]
    mvdat["outcome_beta"] = mvdat["outcome_beta"][instruments]
    mvdat["outcome_se"] = mvdat["outcome_se"][instruments]
    mvdat["outcome_pval"] = mvdat["outcome_pval"][instruments]

    return mv_multiple(
        mvdat,
        intercept=intercept,
        instrument_specific=instrument_specific,
        pval_threshold=pval_threshold,
        plots=plots
    )
