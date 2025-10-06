import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

def mr_grip(b_exp, b_out, se_exp, se_out, parameters=None):
    b_exp = np.asarray(b_exp)
    b_out = np.asarray(b_out)
    se_exp = np.asarray(se_exp)
    se_out = np.asarray(se_out)

    if not (len(b_exp) == len(b_out) == len(se_exp) == len(se_out)):
        raise ValueError("The lengths of b_exp, b_out, se_exp, and se_out must all be equal.")

    nulllist = {"b": None, "se": None, "pval": None, "b.adj": np.nan, "se.adj": np.nan,
                "pval.adj": np.nan, "nsnp": None, "Q": None, "Q_df": None, "Q_pval": None,
                "mod": None, "smod": None, "dat": None}

    valid = ~np.isnan(b_exp) & ~np.isnan(b_out) & ~np.isnan(se_exp) & ~np.isnan(se_out)
    if np.sum(valid) < 3:
        return nulllist

    b_exp = b_exp[valid]
    b_out = b_out[valid]
    se_out = se_out[valid]

    dat = pd.DataFrame({
        "b_out": b_out,
        "b_exp": b_exp,
        "se_out": se_out
    })

    grip_out = b_out * b_exp
    grip_exp = b_exp ** 2
    weights = 1 / (grip_exp * se_out**2)

    X = sm.add_constant(grip_exp)
    model = sm.WLS(grip_out, X, weights=weights).fit()

    b = model.params[1]
    se = model.bse[1]
    df = len(b_exp) - 2
    pval = 2 * t.sf(np.abs(b / se), df)

    return {"b": b, "se": se, "pval": pval, "b.adj": np.nan, "se.adj": np.nan, "pval.adj": np.nan, "nsnp": len(b_exp), "mod": model, "smod": model.summary(), "dat": dat}
