import numpy as np
from scipy import stats
import statsmodels.api as sm
from mr import mr_sign
from rucker import isq, mr_rucker

def system_metrics(dat):
    metrics = {}
    metrics["nsnp"] = len(dat)
    metrics["nout"] = dat["samplesize.outcome"].mean(skipna=True)
    metrics["nexp"] = dat["samplesize.exposure"].mean(skipna=True)

    Fstat = (dat["beta.exposure"]**2 / dat["se.exposure"]**2).replace([np.inf, -np.inf], 300)
    metrics["meanF"] = Fstat.mean(skipna=True)
    metrics["varF"] = Fstat.var(skipna=True)
    metrics["medianF"] = Fstat.median(skipna=True)

    if len(dat) > 1:
        metrics["egger_isq"] = isq(np.abs(dat["beta.exposure"].values), dat["se.exposure"].values)

    if len(dat) > 2:
        sct = mr_sign(
            dat["beta.exposure"].values,
            dat["beta.outcome"].values,
            dat["se.exposure"].values,
            dat["se.outcome"].values
        )
        metrics["sct"] = -np.log10(sct["pval"]) * np.sign(sct["b"])

        ruck = mr_rucker(dat)
        ruck=ruck[0]
        Q = ruck["Q"]
        intercept = ruck["intercept"]

        metrics["Isq"] = (Q["Q"].iloc[0] - (Q["df"].iloc[0] - 1)) / Q["Q"].iloc[0]
        metrics["Isqe"] = (Q["Q"].iloc[1] - (Q["df"].iloc[1] - 1)) / Q["Q"].iloc[1]
        metrics["Qdiff"] = Q["Q"].iloc[2]
        metrics["intercept"] = abs(intercept["Estimate"].iloc[0]) / intercept["SE"].iloc[0]

        # IVW
        lmod_ivw = ruck["lmod_ivw"]
        infl_ivw = lmod_ivw.get_influence()
        dfbetas_ivw = np.abs(infl_ivw.dfbetas)
        cooks_ivw = infl_ivw.cooks_distance[0]

        thresh_dfb = 2 / np.sqrt(len(dat))
        thresh_cooks_ivw = 4 / (len(dat) - 2)

        for i in range(min(3, dfbetas_ivw.shape[1])):
            metrics[f"dfb{i+1}_ivw"] = np.mean(dfbetas_ivw[:, i] > thresh_dfb)
        metrics["cooks_ivw"] = np.mean(cooks_ivw > thresh_cooks_ivw)

        # Egger
        lmod_egger = ruck["lmod_egger"]
        infl_egger = lmod_egger.get_influence()
        dfbetas_egger = np.abs(infl_egger.dfbetas)
        cooks_egger = infl_egger.cooks_distance[0]
        thresh_cooks_egger = 4 / (len(dat) - 3)

        for i in range(min(3, dfbetas_egger.shape[1])):
            metrics[f"dfb{i+1}_egger"] = np.mean(dfbetas_egger[:, i] > thresh_dfb)
        metrics["cooks_egger"] = np.mean(cooks_egger > thresh_cooks_egger)

        # Breusch-Pagan
        exog_ivw = sm.add_constant(lmod_ivw.model.exog, has_constant='add')
        bp_ivw = sm.stats.diagnostic.het_breuschpagan(lmod_ivw.resid, exog_ivw)

        exog_egger = sm.add_constant(lmod_egger.model.exog, has_constant='add')
        bp_egger = sm.stats.diagnostic.het_breuschpagan(lmod_egger.resid, exog_egger)

        metrics["homosc_ivw"] = bp_ivw[0]
        metrics["homosc_egg"] = bp_egger[0]

        metrics["shap_ivw"] = stats.shapiro(lmod_ivw.resid)[0]
        metrics["shap_egger"] = stats.shapiro(lmod_egger.resid)[0]
        metrics["ks_ivw"] = stats.kstest(lmod_ivw.resid, "norm")[0]
        metrics["ks_egger"] = stats.kstest(lmod_egger.resid, "norm")[0]

    return metrics