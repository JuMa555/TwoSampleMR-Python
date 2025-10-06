from scipy.stats import norm
import numpy as np
from add_rsq import add_rsq

def steiger_filtering(dat):
    return dat.groupby(["id.exposure", "id.outcome"], group_keys=False).apply(steiger_filtering_internal).reset_index(drop=True)


def steiger_filtering_internal(dat):
    dat = dat.copy()

    if "units.outcome" not in dat.columns:
        dat["units.outcome"] = np.nan
    if "units.exposure" not in dat.columns:
        dat["units.exposure"] = np.nan

    if dat["exposure"].nunique(dropna=True) != 1:
        raise ValueError(f"Expected 1 unique exposure, found: {dat['exposure'].unique()}")
    if dat["outcome"].nunique(dropna=True) != 1:
        raise ValueError(f"Expected 1 unique outcome, found: {dat['outcome'].unique()}")
    if dat["units.exposure"].dropna().nunique() > 1:
        raise ValueError(f"Multiple units.exposure detected: {dat['units.exposure'].unique()}")
    if dat["units.outcome"].dropna().nunique() > 1:
        raise ValueError(f"Multiple units.outcome detected: {dat['units.outcome'].unique()}")

    dat = add_rsq(dat)

    r1 = np.sqrt(dat["rsq.exposure"])
    r2 = np.sqrt(dat["rsq.outcome"])
    n1 = dat["effective_n.exposure"]
    n2 = dat["effective_n.outcome"]

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se

    dat["steiger_dir"] = dat["rsq.exposure"] > dat["rsq.outcome"]
    dat["steiger_pval"] = 2 * norm.sf(np.abs(z_stat))

    return dat
