import numpy as np
import pandas as pd
from scipy.stats import f
from scipy.optimize import minimize_scalar
from scipy.stats import linregress, chi2
import matplotlib.pyplot as plt
import seaborn as sns


def add_rsq(dat):
    dat = dat.copy()

    if "id.exposure" in dat.columns:
        dat = dat.groupby("id.exposure", group_keys=False).apply(lambda x: add_rsq_one(x, "exposure"))

    if "id.outcome" in dat.columns:
        dat = dat.groupby("id.outcome", group_keys=False).apply(lambda x: add_rsq_one(x, "outcome"))

    return dat.reset_index(drop=True)


def add_rsq_one(dat, what="exposure"):
    dat = dat.copy()
    unit_col = f"units.{what}"
    rsq_col = f"rsq.{what}"

    if unit_col not in dat.columns:
        dat[unit_col] = np.nan

    if dat[what].dropna().nunique() != 1:
        raise ValueError(f"Expected one unique value in '{what}', got: {dat[what].unique()}")

    if dat[unit_col].dropna().nunique() > 1:
        raise ValueError(f"Multiple units found in '{unit_col}': {dat[unit_col].unique()}")
    elif dat[unit_col].dropna().nunique() == 0:
        print(f"Warning: All values in '{unit_col}' are missing; default handling may apply.")

    if rsq_col not in dat.columns:
        pval_col = f"pval.{what}"
        dat.loc[dat[pval_col] < 1e-300, pval_col] = 1e-300

        if compareNA(dat[unit_col].iloc[0], "log odds"):
            prevalence_col = f"prevalence.{what}"
            if prevalence_col not in dat.columns:
                dat[prevalence_col] = 0.1
                print(f"Warning: Assuming {what} prevalence of 0.1. You can add '{prevalence_col}' to override.")

            beta = dat[f"beta.{what}"]
            eaf = dat[f"eaf.{what}"]
            ncase = dat[f"ncase.{what}"]
            ncontrol = dat[f"ncontrol.{what}"]
            prevalence = dat[prevalence_col]

            ind1 = beta.notna() & eaf.notna() & ncase.notna() & ncontrol.notna() & prevalence.notna()
            dat[rsq_col] = np.nan
            if ind1.sum() > 0:
                r = get_r_from_lor(
                    beta[ind1],
                    eaf[ind1],
                    ncase[ind1],
                    ncontrol[ind1],
                    prevalence[ind1]
                )
                dat.loc[ind1, rsq_col] = r**2
                dat.loc[ind1, f"effective_n.{what}"] = effective_n(ncase[ind1], ncontrol[ind1])
            else:
                print("Try adding metadata with add_metadata()")

        elif dat[unit_col].astype(str).str.contains("SD", na=False).all() and dat[f"eaf.{what}"].notna().all():
            beta = dat[f"beta.{what}"]
            eaf = dat[f"eaf.{what}"]
            dat[rsq_col] = 2 * beta**2 * eaf * (1 - eaf)
            dat[f"effective_n.{what}"] = dat[f"samplesize.{what}"]


        else:
            beta = dat[f"beta.{what}"]
            se = dat[f"se.{what}"]
            n = dat[f"samplesize.{what}"]
            ind1 = dat[f"pval.{what}"].notna() & n.notna()
            dat[rsq_col] = np.nan
            if ind1.sum() > 0:
                r = get_r_from_bsen(beta[ind1], se[ind1], n[ind1])
                dat.loc[ind1, rsq_col] = r**2
                dat.loc[ind1, f"effective_n.{what}"] = n[ind1]
            else:
                print("Try adding metadata with add_metadata()")

    return dat


def get_r_from_pn_less_accurate(p, n):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)

    p[p == 1] = 0.999
    p[p == 0] = 1e-200

    q1 = chi2.isf(p, df=1)
    q2 = chi2.isf(p, df=n - 2)
    qval = q1 / (q2 / (n - 2))

    r = np.sqrt(np.sum(qval / (n - qval)))

    if r >= 1:
        print("Warning: Correlation greater than 1, make sure SNPs are pruned for LD.")

    return r


def test_r_from_pn():
    ns = [10, 100, 1000, 10000, 100000]
    rsqs = 10**np.linspace(-4, -0.5, 30)

    results = []

    for n in ns:
        for rsq in rsqs:
            x = np.random.normal(size=n)
            x = (x - np.mean(x)) / np.std(x)

            y_noise = np.random.normal(size=n)
            y_noise = (y_noise - np.mean(y_noise)) / np.std(y_noise)

            y = x * np.sqrt(rsq) + y_noise * np.sqrt(1 - rsq)

            rsq_emp = np.corrcoef(x, y)[0, 1]**2
            slope, intercept, r_value, pval, stderr = linregress(x, y)

            pval = max(pval, 1e-300)

            rsq1 = get_r_from_pn_less_accurate(pval, n)**2
            rsq2 = get_r_from_pn(pval, n)**2

            results.append({
                "n": n,
                "rsq": rsq,
                "rsq_emp": rsq_emp,
                "pval": pval,
                "rsq1": rsq1,
                "rsq2": rsq2
            })

    df = pd.DataFrame(results)
    df_long = df.melt(id_vars=["n", "rsq", "rsq_emp", "pval"], value_vars=["rsq1", "rsq2"],
                      var_name="out", value_name="value")

    g = sns.FacetGrid(df_long, col="n", col_wrap=3, height=4, sharex=False, sharey=False)
    g.map_dataframe(sns.lineplot, x="rsq_emp", y="value", hue="out")
    for ax in g.axes.flat:
        ax.plot([1e-4, 1], [1e-4, 1], linestyle="dotted", color="grey")
        ax.set_xscale("log")
        ax.set_yscale("log")
    g.add_legend()
    g.set_axis_labels("Empirical R²", "Estimated R²")

    plt.tight_layout()
    return {"dat": df_long, "p": g}


def get_r_from_r2n(r2, n):
    Fval = (r2 * (n - 2)) / (1 - r2)
    return f.sf(Fval, 1, n - 1)


def get_r_from_pn(p, n):
    def optim_get_p_from_rn(x, sample_size, pvalue):
        p_est = get_r_from_r2n(x, sample_size)
        return abs(-np.log10(p_est) - -np.log10(pvalue))

    p = np.array(p)
    if np.isscalar(n):
        n = np.full_like(p, n)

    Fval = f.isf(p, 1, n - 1)
    R2 = Fval / (n - 2 + Fval)
    R2 = np.where(np.isfinite(Fval), R2, np.nan)

    for idx in np.where(~np.isfinite(Fval))[0]:
        if p[idx] == 0:
            R2[idx] = np.nan
            print("Warning: P-value of 0 cannot be converted to R value")
        else:
            opt = minimize_scalar(
                optim_get_p_from_rn,
                bounds=(1e-6, 0.999),
                args=(n[idx], p[idx]),
                method="bounded"
            )
            R2[idx] = opt.x if opt.success else np.nan

    return np.sqrt(R2)


def get_r_from_bsen(b, se, n):
    Fval = (b / se) ** 2
    R2 = Fval / (n - 2 + Fval)
    return np.sqrt(R2) * np.sign(b)


def compareNA(v1, v2):
    if pd.isna(v1) and pd.isna(v2):
        return True
    return v1 == v2


def get_r_from_lor(lor, af, ncase, ncontrol, prevalence, model="logit", correction=False):
    lor = np.asarray(lor)
    af = np.asarray(af)
    ncase = np.asarray(ncase)
    ncontrol = np.asarray(ncontrol)
    prevalence = np.asarray(prevalence)

    if lor.shape != af.shape:
        raise ValueError("lor and af must be the same length.")
    if ncase.size == 1:
        ncase = np.full_like(lor, ncase)
    if ncontrol.size == 1:
        ncontrol = np.full_like(lor, ncontrol)
    if prevalence.size == 1:
        prevalence = np.full_like(lor, prevalence)

    nsnp = len(lor)
    r = np.full(nsnp, np.nan)

    for i in range(nsnp):
        if model == "logit":
            ve = np.pi**2 / 3
        elif model == "probit":
            ve = 1
        else:
            raise ValueError("Model must be 'logit' or 'probit'.")

        pop_af = get_population_allele_frequency(
            af[i],
            ncase[i] / (ncase[i] + ncontrol[i]),
            np.exp(lor[i]),
            prevalence[i]
        )
        vg = lor[i]**2 * pop_af * (1 - pop_af)
        r2 = vg / (vg + ve)

        if correction:
            r2 = r2 / 0.58

        r[i] = np.sqrt(r2) * np.sign(lor[i])

    return r


def contingency(af, prop, odds_ratio, eps=1e-15):
    a = odds_ratio - 1
    b = (af + prop) * (1 - odds_ratio) - 1
    c = odds_ratio * af * prop

    if abs(a) < eps:
        z = np.array([-c / b])
    else:
        d = b**2 - 4 * a * c
        if d < eps**2:
            z = np.array([0.0])
        else:
            sqrt_d = np.sqrt(max(0, d))
            z = np.array([(-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a)])

    matrices = []
    for zi in z:
        m = np.array([
            [zi, prop - zi],
            [af - zi, 1 + zi - af - prop]
        ])
        if np.all(m >= 0):
            matrices.append(m)

    if not matrices:
        raise ValueError("No valid contingency matrix with non-negative entries.")

    return matrices[0] if len(matrices) == 1 else matrices


def allele_frequency(g):
    g = np.asarray(g)
    valid = ~np.isnan(g)
    af = (np.sum(g[valid] == 1) + 2 * np.sum(g[valid] == 2)) / (2 * np.sum(valid))
    return af


def get_population_allele_frequency(af, prop, odds_ratio, prevalence):
    af = np.asarray(af)
    prop = np.asarray(prop)
    odds_ratio = np.asarray(odds_ratio)
    prevalence = np.asarray(prevalence)

    if not (len(af) == len(odds_ratio) == len(prop)):
        raise ValueError("Inputs af, prop, and odds_ratio must have the same length.")

    af_result = np.copy(af)
    for i in range(len(odds_ratio)):
        co = contingency(af[i], prop[i], odds_ratio[i])
        af_controls = co[0, 1] / (co[0, 1] + co[1, 1])
        af_cases = co[0, 0] / (co[0, 0] + co[1, 0])
        af_result[i] = af_controls * (1 - prevalence[i]) + af_cases * prevalence[i]

    return af_result


def effective_n(ncase, ncontrol):
    return 2 / (1 / ncase + 1 / ncontrol)