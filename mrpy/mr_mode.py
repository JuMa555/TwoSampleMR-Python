import numpy as np
import pandas as pd
from scipy.stats import norm, chi2, t
from scipy.stats import gaussian_kde


def mr_mode(dat, parameters, mode_method="all"):
    if parameters is None:
        parameters = default_parameters()

    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]
    
    if len(dat) < 3:
        print("Warning: Need at least 3 SNPs")
        return None
    
    b_exp = dat["beta_exposure"].values
    b_out = dat["beta_outcome"].values
    se_exp = dat["se_exposure"].values
    se_out = dat["se_outcome"].values

    def beta(BetaIV_in, seBetaIV_in, phi, use_weights=False, bandwidth_reference=None):
        n = len(BetaIV_in)
        ref = bandwidth_reference if bandwidth_reference is not None else BetaIV_in
        std = np.std(ref)
        mad = np.median(np.abs(ref - np.median(ref)))
        s = 0.9 * min(std, mad / 0.6745) / n ** (1/5)

        weights = (1 / seBetaIV_in**2)
        weights = weights / np.sum(weights)

        beta_vals = []
        for cur_phi in phi:
            h = max(1e-8, s * cur_phi)

            x_grid = np.linspace(min(BetaIV_in) - 3*h, max(BetaIV_in) + 3*h, 1000)
            density = np.zeros_like(x_grid)

            for i in range(n):
                w = weights[i] if use_weights else 1 / n
                density += w * norm.pdf(x_grid, loc=BetaIV_in[i], scale=h)

            beta_vals.append(x_grid[np.argmax(density)])

        return np.array(beta_vals)


    def boot(BetaIV_in, seBetaIV_in, beta_Mode_in, nboot, phi, parameters):
        beta_boot = np.zeros((nboot, len(beta_Mode_in)))
        
        for i in range(nboot):
            BetaIV_boot = np.random.normal(BetaIV_in, seBetaIV_in[:, 0])
            BetaIV_boot_NOME = np.random.normal(BetaIV_in, seBetaIV_in[:, 1])

            beta_boot[i, :len(phi)] = beta(BetaIV_boot, np.ones(len(BetaIV_in)), phi)
            beta_boot[i, len(phi):2*len(phi)] = beta(BetaIV_boot, seBetaIV_in[:, 0], phi)
            
            weights = 1 / seBetaIV_in[:, 0]**2
            penalty = chi2.sf(weights * (BetaIV_boot - beta_boot[i, len(phi):2*len(phi)])**2, df=1)
            pen_weights = weights * np.minimum(1, penalty * parameters["penk"])
            beta_boot[i, 2*len(phi):3*len(phi)] = beta(BetaIV_boot, np.sqrt(1 / pen_weights), phi)

            beta_boot[i, 3*len(phi):4*len(phi)] = beta(BetaIV_boot_NOME, np.ones(len(BetaIV_in)), phi)
            beta_boot[i, 4*len(phi):5*len(phi)] = beta(BetaIV_boot_NOME, seBetaIV_in[:, 1], phi)
        
        return beta_boot
    
    def compute_se_std(beta_boot):
        return np.std(beta_boot, axis=0, ddof=1)


    phi = np.atleast_1d(parameters["phi"])
    nboot = parameters["nboot"]
    alpha = parameters["alpha"]

    BetaIV = b_out / b_exp
    seBetaIV = np.column_stack([
        np.sqrt((se_out**2) / (b_exp**2) + ((b_out**2) * (se_exp**2)) / (b_exp**4)),
        se_out / np.abs(b_exp)
    ])

    beta_SimpleMode = beta(BetaIV, np.ones(len(BetaIV)), phi, use_weights=False)
    beta_WeightedMode = beta(BetaIV, seBetaIV[:, 0], phi, use_weights=True)


    weights = 1 / seBetaIV[:, 0]**2
    penalty = 1 - chi2.cdf(weights * (BetaIV - beta_WeightedMode)**2, df=1)
    pen_weights = weights * np.minimum(1, penalty * parameters["penk"])
    beta_PenalisedMode = beta(BetaIV, np.sqrt(1 / pen_weights), phi, use_weights=True)

    beta_WeightedMode_NOME = beta(BetaIV, seBetaIV[:, 1], phi, use_weights=True)

    beta_Mode = np.concatenate([beta_SimpleMode, beta_WeightedMode, beta_PenalisedMode, beta_SimpleMode, beta_WeightedMode_NOME])

    beta_Mode_boot = boot(BetaIV, seBetaIV, beta_Mode, nboot, phi, parameters)
    se_Mode = compute_se_std(beta_Mode_boot)


    CI_low_Mode = beta_Mode - norm.ppf(1 - alpha / 2) * se_Mode
    CI_upp_Mode = beta_Mode + norm.ppf(1 - alpha / 2) * se_Mode
    
    P_Mode = t.sf(np.abs(beta_Mode / se_Mode), df=len(b_exp) - 1) * 2

    method_labels = np.repeat(
        ["Simple mode", "Weighted mode", "Penalised mode", "Simple mode (NOME)", "Weighted mode (NOME)"], len(phi)
    )


    id_exposure = dat["id_exposure"].iloc[0] if "id_exposure" in dat.columns else ""
    id_outcome = dat["id_outcome"].iloc[0] if "id_outcome" in dat.columns else ""

    Results = pd.DataFrame({
        "id_exposure": id_exposure,
        "id_outcome": id_outcome,
        "method": method_labels,
        "nsnp": len(b_exp),
        "b": beta_Mode,
        "se": se_Mode,
        "ci_low": CI_low_Mode,
        "ci_upp": CI_upp_Mode,
        "pval": P_Mode
    })

    if mode_method == "all":
        return Results
    else:
        if mode_method not in Results["method"].values:
            raise ValueError(f"Mode method '{mode_method}' not found in results.")
        
    selected = Results[Results["method"] == mode_method].iloc[0]
    return {
        "b": selected["b"],
        "se": selected["se"],
        "pval": selected["pval"],
        "nsnp": len(b_exp)
    }

def mr_weighted_mode(b_exp, b_out, se_exp, se_out, parameters=None):
    parameters = default_parameters()

    index = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    
    if np.sum(index) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    b_exp = b_exp[index]
    b_out = b_out[index]
    se_exp = se_exp[index]
    se_out = se_out[index]

    data = pd.DataFrame({
        "beta_exposure": b_exp,
        "beta_outcome": b_out,
        "se_exposure": se_exp,
        "se_outcome": se_out
    })

    return mr_mode(data, parameters=parameters, mode_method="Weighted mode")


def mr_simple_mode(b_exp, b_out, se_exp, se_out, parameters=None):
    parameters = default_parameters()
        
    index = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    
    if np.sum(index) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    b_exp = b_exp[index]
    b_out = b_out[index]
    se_exp = se_exp[index]
    se_out = se_out[index]

    data = pd.DataFrame({
        "beta_exposure": b_exp,
        "beta_outcome": b_out,
        "se_exposure": se_exp,
        "se_outcome": se_out
    })

    return mr_mode(data, parameters=parameters, mode_method="Simple mode")


def mr_weighted_mode_nome(b_exp, b_out, se_exp, se_out, parameters=None):
    if parameters is None:
        parameters = default_parameters()
    
    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    dat = pd.DataFrame({
        "beta_exposure": b_exp[valid],
        "beta_outcome": b_out[valid],
        "se_exposure": se_exp[valid],
        "se_outcome": se_out[valid]
    })
    
    return mr_mode(dat, parameters, mode_method="Weighted mode (NOME)")


def mr_simple_mode_nome(b_exp, b_out, se_exp, se_out, parameters=None):
    if parameters is None:
        parameters = default_parameters()
    
    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    dat = pd.DataFrame({
        "beta_exposure": b_exp[valid],
        "beta_outcome": b_out[valid],
        "se_exposure": se_exp[valid],
        "se_outcome": se_out[valid]
    })
    
    return mr_mode(dat, parameters, mode_method="Simple mode (NOME)")


def default_parameters():

    return {
        "test_dist": "z",
        "nboot": 1000,
        "Cov": 0,
        "penk": 20,
        "phi": [1],
        "alpha": 0.05,
        "Qthresh": 0.05,
        "over_dispersion": True,
        "loss_function": "huber",
        "shrinkage": False
    }