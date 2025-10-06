import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, chi2, binom_test
import statsmodels.api as sm
from mr_mode import mr_simple_mode, mr_weighted_mode, mr_weighted_mode_nome, mr_simple_mode_nome
from mr_grip import mr_grip


def mr(dat, parameters=None, method_list=None):
    if parameters is None:
        parameters = default_parameters()

    method_l = mr_method_list()
    method_info_map = {m["obj"]: m["name"] for m in method_l}
    if method_list is None:
        method_list = [m["obj"] for m in method_l if m["use_by_default"]]

    if not isinstance(dat, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    results = []

    grouped = dat.groupby(["id.exposure", "id.outcome"])
    
    for (exposure_id, outcome_id), group in grouped:
        filtered_group = group[group["mr_keep"]]
        if filtered_group.empty:
            print(f"No SNPs available for MR analysis of '{exposure_id}' on '{outcome_id}'")
            continue

        print(f"Analysing '{exposure_id}' on '{outcome_id}'")

        b_exp = filtered_group["beta.exposure"].values
        b_out = filtered_group["beta.outcome"].values
        se_exp = filtered_group["se.exposure"].values
        se_out = filtered_group["se.outcome"].values

        exposure = filtered_group["exposure"].iloc[0]
        outcome = filtered_group["outcome"].iloc[0]

        for method_name in method_list:
            try:
                method_func = method_map[method_name]
                if method_func is None:
                    raise ValueError(f"Method {method_name} not implemented.")
                
                res = method_func(b_exp, b_out, se_exp, se_out, parameters)
                res.update({
                    "id.exposure": exposure_id,
                    "id.outcome": outcome_id,
                    "exposure": exposure,
                    "outcome": outcome,
                    "method": method_info_map.get(method_name, method_name)
                })
                results.append(res)
            except Exception as e:
                print(f"Error in method {method_name} for '{exposure_id}' on '{outcome_id}': {e}")

    results_df = pd.DataFrame(results)

    results_df = results_df[[
        "id.exposure", "id.outcome", "outcome", "exposure", "method", 
        "nsnp", "b", "se", "pval"
    ]]
    results_df = results_df.sort_values(by=["b"], ascending=True)

    return results_df


def mr_method_list():
    return [
        {"obj": "mr_wald_ratio", "name": "Wald ratio", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_meta_fixed_simple", "name": "Fixed effects meta analysis (simple SE)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_meta_fixed", "name": "Fixed effects meta analysis (delta method)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": True},
        {"obj": "mr_meta_random", "name": "Random effects meta analysis (delta method)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": True},
        {"obj": "mr_two_sample_ml", "name": "Maximum likelihood", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": True},
        {"obj": "mr_egger_regression", "name": "MR Egger", "PubmedID": "26050253", "Description": "", "use_by_default": True, "heterogeneity_test": True},
        {"obj": "mr_egger_regression_bootstrap", "name": "MR Egger (bootstrap)", "PubmedID": "26050253", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_simple_median", "name": "Simple median", "PubmedID": "27061298", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_weighted_median", "name": "Weighted median", "PubmedID": "27061298", "Description": "", "use_by_default": True, "heterogeneity_test": False},
        {"obj": "mr_penalised_weighted_median", "name": "Penalised weighted median", "PubmedID": "27061298", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_ivw", "name": "Inverse variance weighted", "PubmedID": "24114802", "Description": "", "use_by_default": True, "heterogeneity_test": True},
    #   {"obj": "mr_ivw_radial", "name": "IVW radial", "PubmedID": "29961852", "Description": "", "use_by_default": False, "heterogeneity_test": True},
        {"obj": "mr_ivw_mre", "name": "Inverse variance weighted (multiplicative random effects)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_ivw_fe", "name": "Inverse variance weighted (fixed effects)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_simple_mode", "name": "Simple mode", "PubmedID": "29040600", "Description": "", "use_by_default": True, "heterogeneity_test": False},
        {"obj": "mr_weighted_mode", "name": "Weighted mode", "PubmedID": "", "Description": "", "use_by_default": True, "heterogeneity_test": False},
        {"obj": "mr_weighted_mode_nome", "name": "Weighted mode (NOME)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_simple_mode_nome", "name": "Simple mode (NOME)", "PubmedID": "29040600", "Description": "", "use_by_default": False, "heterogeneity_test": False},
    #    {"obj": "mr_raps", "name": "Robust adjusted profile score (RAPS)", "PubmedID": "", "Description": "", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_sign", "name": "Sign concordance test", "PubmedID": "", "Description": "Tests for concordance of signs between exposure and outcome", "use_by_default": False, "heterogeneity_test": False},
        {"obj": "mr_uwr", "name": "Unweighted regression", "PubmedID": "", "Description": "Doesn\"t use any weights", "use_by_default": False, "heterogeneity_test": True},
   		{"obj": "mr_grip", "name": "MR GRIP", "PubmedID": "", "Description": "Allele coding invariant regression", "use_by_default": False, "heterogeneity_test":  False}
    ]


def default_parameters():
    return {
        "test_dist": "z",
        "nboot": 1000,
        "Cov": 0,
        "penk": 20,
        "phi": 1,
        "alpha": 0.05,
        "Qthresh": 0.05,
        "over_dispersion": True,
        "loss_function": "huber",
        "shrinkage": False
    }


def mr_wald_ratio(b_exp, b_out, se_exp, se_out, parameters):
    if len(b_exp) > 1:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    b = b_out[0] / b_exp[0]
    se = se_out[0] / abs(b_exp[0])
    z = abs(b) / se
    pval = 2 * (1 - stats.norm.cdf(z))
    
    return {"b": b, "se": se, "pval": pval, "nsnp": 1}


def mr_meta_fixed_simple(b_exp, b_out, se_exp, se_out, parameters):
    mask = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(mask) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}
    
    b_exp, b_out, se_out = np.array(b_exp)[mask], np.array(b_out)[mask], np.array(se_out)[mask]
    
    b = np.sum(b_exp * b_out / se_out**2) / np.sum(b_exp**2 / se_out**2)
    se = np.sqrt(1 / np.sum(b_exp**2 / se_out**2))
    pval = 2 * norm.sf(abs(b) / se)
    
    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp)}


def mr_meta_fixed(b_exp, b_out, se_exp, se_out, parameters):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 1:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}
    
    ratio = b_out / b_exp
    ratio_se = np.sqrt((se_out**2 / b_exp**2) + (b_out**2 / b_exp**4) * se_exp**2 - 2 * (b_out / b_exp**3) * parameters["Cov"])

    weights = 1 / ratio_se**2
    b_fixed = np.sum(weights * ratio) / np.sum(weights)
    se_fixed = np.sqrt(1 / np.sum(weights))
    pval_fixed = chi2.sf((b_fixed / se_fixed)**2, 1)

    Q = np.sum(weights * (ratio - b_fixed)**2)
    df_Q = len(ratio) - 1
    Q_pval = chi2.sf(Q, df_Q)

    return {"b": b_fixed, "se": se_fixed, "pval": pval_fixed, "nsnp": len(b_exp), "Q": Q, "Q_df": df_Q, "Q_pval": Q_pval}


def mr_meta_random(b_exp, b_out, se_exp, se_out, parameters):
    if sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 2:
        return {"b": None, "se": None, "pval": None, "nsnp": None, "Q": None, "Q_df": None, "Q_pval": None}
    
    ratio = b_out / b_exp
    ratio_se = np.sqrt((se_out**2 / b_exp**2) + (b_out**2 / b_exp**4) * se_exp**2 - 2 * (b_out / b_exp**3) * parameters["Cov"])

    weights = 1 / ratio_se**2
    weighted_mean = np.sum(weights * ratio) / np.sum(weights)
    Q = np.sum(weights * (ratio - weighted_mean)**2)
    Q_df = len(ratio) - 1
    Q_pval = chi2.sf(Q, Q_df)

    tau_squared = max(0, (Q - Q_df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
    weights_random = 1 / (ratio_se**2 + tau_squared)
    b_random = np.sum(weights_random * ratio) / np.sum(weights_random)
    se_random = np.sqrt(1 / np.sum(weights_random))
    pval_random = chi2.sf((b_random / se_random)**2, 1)

    return {"b": b_random, "se": se_random, "pval": pval_random, "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}


def mr_two_sample_ml(b_exp, b_out, se_exp, se_out, parameters):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}

    def loglikelihood(param):
        return ( 0.5 * np.sum((b_exp - param[:len(b_exp)]) ** 2 / se_exp ** 2) + 0.5 * np.sum((b_out - param[len(b_exp)] * param[:len(b_exp)]) ** 2 / se_out ** 2) )

    initial_guess = np.concatenate([
        b_exp,
        [np.sum(b_exp * b_out / se_out ** 2) / np.sum(b_exp ** 2 / se_out ** 2)]
    ])

    try:
        opt = minimize(
            loglikelihood,
            initial_guess,
            method="BFGS",
            options={"maxiter": 25000}
        )
    except Exception as e:
        print("mr_two_sample_ml failed to converge")
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}

    b = opt.x[len(b_exp)]
    try:
        hessian_inv = opt.hess_inv
        se = np.sqrt(hessian_inv[len(b_exp), len(b_exp)])
    except Exception as e:
        print("mr_two_sample_ml failed to converge")
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}

    pval = 2 * norm.sf(abs(b) / se)

    Q = 2 * opt.fun
    Q_df = len(b_exp) - 1
    Q_pval = chi2.sf(Q, Q_df)

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}


def mr_egger_regression(b_exp, b_out, se_exp, se_out, parameters):
    if len(b_exp) != len(b_out) or len(se_exp) != len(se_out) or len(b_exp) != len(se_out):
        raise ValueError("All input arrays must have the same length.")

    nulllist = {
        "b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan,
        "b_i": np.nan, "se_i": np.nan, "pval_i": np.nan,
        "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan,
        "mod": None, "dat": None
    }

    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 3:
        return nulllist

    def sign0(x):
        x = np.array(x)
        x[x == 0] = 1
        return np.sign(x)

    to_flip = sign0(b_exp) == -1
    b_out = b_out * sign0(b_exp)
    b_exp = np.abs(b_exp)

    dat = pd.DataFrame({"b_out": b_out, "b_exp": b_exp, "se_exp": se_exp, "se_out": se_out, "flipped": to_flip})

    weights = 1 / se_out**2
    X = sm.add_constant(b_exp)
    model = sm.WLS(b_out, X, weights=weights).fit()
    
    if len(model.params) > 1:
        b = model.params[1]
        se = model.bse[1] / min(1, model.mse_resid**0.5)

        pval = 2 * stats.t.sf(np.abs(b / se), df=len(b_exp) - 2)

        b_i = model.params[0]
        se_i = model.bse[0] / min(1, model.mse_resid**0.5)
        pval_i = 2 * stats.t.sf(np.abs(b_i / se_i), df=len(b_exp) - 2)

        Q_df = len(b_exp) - 2
        Q = model.mse_resid * Q_df
        Q_pval = stats.chi2.sf(Q, Q_df)

    else:
        print("Warning: Collinearities in MR Egger, try LD pruning the exposure variables.")
        return nulllist

    return {
        "b": b, "se": se, "pval": pval, "nsnp": len(b_exp),
        "b_i": b_i, "se_i": se_i, "pval_i": pval_i,
        "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval,
        "mod": model, "dat": dat
    }


def linreg(x, y, w=None):
    if w is None:
        w = np.ones_like(x)
    
    xp = w * x
    yp = w * y
    
    bhat = np.cov(x * w, y * w)[0, 1] / np.var(x * w, ddof=1)
    ahat = np.nanmean(y) - np.nanmean(x) * bhat
    yhat = ahat + bhat * x
    
    se = np.sqrt(np.sum((yp - yhat)**2) / (np.sum(~np.isnan(yhat)) - 2) / np.dot(x.T, x))
    
    residual_sum_of_squares = np.sum(w * (y - yhat)**2)
    se = np.sqrt(residual_sum_of_squares / (np.sum(~np.isnan(yhat)) - 2) / np.sum(w * x**2))
    
    pval = 2 * norm.sf(abs(bhat / se))
    
    return {"ahat": ahat, "bhat": bhat, "se": se, "pval": pval}


def mr_egger_regression_bootstrap(b_exp, b_out, se_exp, se_out, parameters):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 3:
        return {
            "b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan,
            "b_i": np.nan, "se_i": np.nan, "pval_i": np.nan,
            "mod": None, "smod": None, "dat": None
        }
    
    nboot = parameters["nboot"]
    res = np.zeros((nboot + 1, 2))
    
    for i in range(nboot):
        xs = np.random.normal(b_exp, se_exp)
        ys = np.random.normal(b_out, se_out)

        ys *= np.sign(xs)
        xs = np.abs(xs)

        weights = 1 / se_out**2
        r = linreg(xs, ys, weights)

        res[i, 0] = r["ahat"]
        res[i, 1] = r["bhat"]
    b = np.mean(res[:, 1], axis=0)
    se = np.std(res[:, 1], axis=0)
    pval = np.sum(np.sign(b) * res[:, 1] < 0) / nboot
    b_i = np.mean(res[:, 0], axis=0)
    se_i = np.std(res[:, 0], axis=0)
    pval_i = np.sum(np.sign(b_i) * res[:, 0] < 0) / nboot

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp), "b_i": b_i, "se_i": se_i, "pval_i": pval_i}


def mr_weighted_median(b_exp, b_out, se_exp, se_out, parameters=None):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    b_iv = b_out / b_exp

    VBj = (se_out**2) / b_exp**2 + (b_out**2) * (se_exp**2) / b_exp**4
    weights = 1 / VBj

    b = weighted_median(b_iv, weights)

    nboot = parameters.get("nboot", 1000)
    se = weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, weights, nboot)

    pval = 2 * stats.norm.sf(abs(b / se))

    return {"b": b, "se": se, "pval": pval, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan, "nsnp": len(b_exp)}


def mr_simple_median(b_exp, b_out, se_exp, se_out, parameters=None):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    b_iv = b_out / b_exp

    weights = np.full_like(b_exp, 1 / len(b_exp), dtype=np.float64)

    b = weighted_median(b_iv, weights)

    nboot = parameters.get("nboot", 1000)
    se = weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, weights, nboot)

    pval = 2 * stats.norm.sf(abs(b / se))

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp)}


def weighted_median(b_iv, weights):
    sorted_indices = np.argsort(b_iv)
    values_sorted = b_iv[sorted_indices]
    weights_sorted = weights[sorted_indices]
    cumulative_weights = np.cumsum(weights_sorted) - 0.5 * weights_sorted
    cumulative_weights /= np.sum(weights_sorted)

    below_index = np.max(np.where(cumulative_weights < 0.5))
    if below_index + 1 < len(values_sorted):
        b = values_sorted[below_index] + (values_sorted[below_index + 1] - values_sorted[below_index]) * \
            (0.5 - cumulative_weights[below_index]) / (cumulative_weights[below_index + 1] - cumulative_weights[below_index])
    else:
        b = values_sorted[below_index]

    return b


def weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, weights, nboot=1000):
    medians = np.zeros(nboot)
    for i in range(nboot):
        b_exp_boot = np.random.normal(loc=b_exp, scale=se_exp, size=len(b_exp))
        b_out_boot = np.random.normal(loc=b_out, scale=se_out, size=len(b_out))
        beta_iv_boot = b_out_boot / b_exp_boot
        medians[i] = weighted_median(beta_iv_boot, weights)

    return np.std(medians, ddof=1)


def mr_penalised_weighted_median(b_exp, b_out, se_exp, se_out, parameters=None):
    if parameters is None:
        parameters = default_parameters()

    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 3:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    b_exp = np.array(b_exp)[valid]
    b_out = np.array(b_out)[valid]
    se_exp = np.array(se_exp)[valid]
    se_out = np.array(se_out)[valid]

    betaIV = b_out / b_exp
    betaIVW = np.sum(b_out * b_exp / se_out**2) / np.sum(b_exp**2 / se_out**2)
    VBj = (se_out**2) / (b_exp**2) + (b_out**2) * (se_exp**2) / (b_exp**4)
    weights = 1 / VBj

    bwm = mr_weighted_median(b_exp, b_out, se_exp, se_out, parameters)
    penalty = chi2.sf(weights * (betaIV - bwm["b"])**2, df=1)
    pen_weights = weights * np.minimum(1, penalty * parameters["penk"])

    b = weighted_median(betaIV, pen_weights)
    se = weighted_median_bootstrap(b_exp, b_out, se_exp, se_out, pen_weights, parameters["nboot"])
    pval = 2 * norm.sf(abs(b / se))

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp)}


def mr_median(dat, parameters=default_parameters()):
    if "mr_keep" in dat.columns:
        dat = dat[dat["mr_keep"]]

    if len(dat) < 3:
        print("Need at least 3 SNPs")
        return None

    b_exp = dat["beta.exposure"]
    b_out = dat["beta.outcome"]
    se_exp = dat["se.exposure"]
    se_out = dat["se.outcome"]

    sm = mr_simple_median(b_exp, b_out, se_exp, se_out, parameters)
    wm = mr_weighted_median(b_exp, b_out, se_exp, se_out, parameters)
    pm = mr_penalised_weighted_median(b_exp, b_out, se_exp, se_out, parameters)

    res = pd.DataFrame({
        "id.exposure": [dat["id.exposure"].iloc[0]] * 3,
        "id.outcome": [dat["id.outcome"].iloc[0]] * 3,
        "method": ["Simple median", "Weighted median", "Penalised median"],
        "nsnp": [len(b_exp)] * 3,
        "b": [sm["b"], wm["b"], pm["b"]],
        "se": [sm["se"], wm["se"], pm["se"]],
    })

    res["ci_low"] = res["b"] - stats.norm.ppf(1 - parameters["alpha"] / 2) * res["se"]
    res["ci_upp"] = res["b"] + stats.norm.ppf(1 - parameters["alpha"] / 2) * res["se"]
    res["pval"] = [sm["pval"], wm["pval"], pm["pval"]]
    return res


def mr_ivw(b_exp, b_out, se_exp, se_out, parameters=None):
    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    weights = 1 / se_out**2
    X = b_exp.reshape(-1, 1)
    y = b_out
    model = sm.WLS(y, X, weights=weights).fit()

    b = model.params[0]
    se = model.bse[0] / min(1, model.mse_resid**0.5)

    z = abs(b / se)
    pval = 2 * stats.norm.sf(z)

    Q_df = len(b_exp) - 1
    Q = model.mse_resid * Q_df
    Q_pval = stats.chi2.sf(Q, Q_df)

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}


def mr_uwr(b_exp, b_out, se_exp, se_out, parameters=None):
    import numpy as np
    from scipy.stats import norm, chi2

    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    X = np.array(b_exp).reshape(-1, 1)
    y = np.array(b_out)
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    b = coef[0]

    sigma = np.sqrt(residuals / (len(b_exp) - 1)) if len(b_exp) > 1 else 1

    residual_variance = np.sum((y - X.dot(coef))**2) / (len(b_exp) - 1)

    se = np.sqrt(residual_variance / np.sum((X - X.mean())**2)) / min(1, sigma)
    pval = 2 * norm.sf(abs(b / se))

    Q_df = len(b_exp) - 1
    Q = sigma**2 * Q_df
    Q_pval = chi2.sf(Q, Q_df)

    return {"b": b, "se": se[0], "pval": pval[0], "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}


def mr_ivw_mre(b_exp, b_out, se_exp, se_out, parameters=None):
    b_exp = np.asarray(b_exp)
    b_out = np.asarray(b_out)
    se_exp = np.asarray(se_exp)
    se_out = np.asarray(se_out)

    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}

    b_exp = b_exp[valid]
    b_out = b_out[valid]
    se_out = se_out[valid]
    weights = 1 / se_out**2

    X = b_exp.reshape(-1, 1)
    y = b_out
    W = np.diag(weights)

    XTWX = X.T @ W @ X
    XTWY = X.T @ W @ y

    b = np.linalg.solve(XTWX, XTWY)
    residuals = y - X @ b
    Q_df = len(b_exp) - 1
    sigma2 = (residuals.T @ W @ residuals) / Q_df

    se = np.sqrt(sigma2 / XTWX[0, 0])
    pval = 2 * norm.sf(abs(b / se))

    Q = sigma2 * Q_df
    Q_pval = chi2.sf(Q, Q_df)

    return {"b": b[0], "se": se, "pval": pval[0], "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}


def mr_ivw_fe(b_exp, b_out, se_exp, se_out, parameters=None):
    b_exp = np.asarray(b_exp)
    b_out = np.asarray(b_out)
    se_exp = np.asarray(se_exp)
    se_out = np.asarray(se_out)

    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 2:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan, "Q": np.nan, "Q_df": np.nan, "Q_pval": np.nan}

    b_exp = b_exp[valid]
    b_out = b_out[valid]
    se_out = se_out[valid]

    weights = 1 / se_out**2
    X = b_exp.reshape(-1, 1)
    y = b_out

    model = sm.WLS(y, X, weights=weights).fit()
    b = model.params[0]
    sigma = np.sqrt(model.mse_resid)
    se = model.bse[0] / sigma
    pval = 2 * norm.sf(abs(b / se))

    Q_df = len(b_exp) - 1
    Q = sigma**2 * Q_df
    Q_pval = chi2.sf(Q, Q_df)

    return {"b": b, "se": se, "pval": pval, "nsnp": len(b_exp), "Q": Q, "Q_df": Q_df, "Q_pval": Q_pval}



def mr_sign(b_exp, b_out, se_exp=None, se_out=None, parameters=None):
    b_exp = np.where(b_exp == 0, np.nan, b_exp)
    b_out = np.where(b_out == 0, np.nan, b_out)

    if np.sum(~np.isnan(b_exp) & ~np.isnan(b_out)) < 6:
        return {"b": np.nan, "se": np.nan, "pval": np.nan, "nsnp": np.nan}

    x = np.sum(np.sign(b_exp) == np.sign(b_out))
    n = np.sum(~np.isnan(b_exp) & ~np.isnan(b_out))

    out = binom_test(x=x, n=n, p=0.5, alternative="two-sided")
    b = (x / n - 0.5) * 2

    return {"b": b, "se": np.nan, "pval": out, "nsnp": n}


method_map = {
    "mr_wald_ratio": mr_wald_ratio,
    "mr_meta_fixed_simple": mr_meta_fixed_simple,
    "mr_meta_fixed": mr_meta_fixed,
    "mr_meta_random": mr_meta_random,
    "mr_two_sample_ml": mr_two_sample_ml,
    "mr_egger_regression": mr_egger_regression,
    "mr_egger_regression_bootstrap": mr_egger_regression_bootstrap,
    "mr_simple_median": mr_simple_median,
    "mr_weighted_median": mr_weighted_median,
    "mr_penalised_weighted_median": mr_penalised_weighted_median,
    "mr_ivw": mr_ivw,
    "mr_uwr": mr_uwr,
    "mr_ivw_mre": mr_ivw_mre,
    "mr_ivw_fe" : mr_ivw_fe,
    "mr_simple_mode": mr_simple_mode,
    "mr_weighted_mode": mr_weighted_mode,
    "mr_weighted_mode_nome": mr_weighted_mode_nome,
    "mr_simple_mode_nome": mr_simple_mode_nome,
#    "mr_raps": mr_raps,
    "mr_sign": mr_sign,
    "mr_grip": mr_grip
}
