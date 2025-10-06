import numpy as np
import pandas as pd
from mr import mr_egger_regression, mr_egger_regression_bootstrap, default_parameters
import seaborn as sns
import matplotlib.pyplot as plt

def mr_scatter_plot(mr_results, dat):
    plots = {}

    grouped = dat.groupby(["id.exposure", "id.outcome"])

    for (id_exp, id_out), group in grouped:
        if len(group) < 2 or group["mr_keep"].sum() == 0:
            print(f"Insufficient number of SNPs for {id_exp} on {id_out}")
            continue

        d = group[group["mr_keep"]].copy()

        index = d["beta.exposure"] < 0
        d.loc[index, "beta.exposure"] *= -1
        d.loc[index, "beta.outcome"] *= -1

        mrres = mr_results[(mr_results["id.exposure"] == id_exp) & (mr_results["id.outcome"] == id_out)].copy()
        mrres["a"] = 0

        if "MR Egger" in mrres["method"].values:
            temp = mr_egger_regression(
                d["beta.exposure"].values,
                d["beta.outcome"].values,
                d["se.exposure"].values,
                d["se.outcome"].values,
                default_parameters()
            )
            mrres.loc[mrres["method"] == "MR Egger", "a"] = temp["b_i"]

        if "MR Egger (bootstrap)" in mrres["method"].values:
            temp = mr_egger_regression_bootstrap(
                d["beta.exposure"].values,
                d["beta.outcome"].values,
                d["se.exposure"].values,
                d["se.outcome"].values,
                default_parameters()
            )
            mrres.loc[mrres["method"] == "MR Egger (bootstrap)", "a"] = temp["b_i"]

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.errorbar(d["beta.exposure"], d["beta.outcome"],
                    xerr=d["se.exposure"], yerr=d["se.outcome"],
                    fmt='o', color='black', ecolor='grey', alpha=0.7)
        
        palette = sns.color_palette("tab10", n_colors=20)
        method_colors = dict(zip(mrres["method"], palette))

        palette = sns.color_palette("tab10", n_colors=len(mrres))
        for i, (_, row) in enumerate(mrres.iterrows()):
            slope = row["b"]
            intercept = row["a"]
            method = row["method"]
            color = method_colors.get(method, "black")
            x_vals = np.array([d["beta.exposure"].min(), d["beta.exposure"].max()])
            y_vals = intercept + slope * x_vals
            ax.plot(x_vals, y_vals, label=method, color=color)

        ax.set_xlabel(f"SNP effect on {d['exposure'].iloc[0]}")
        ax.set_ylabel(f"SNP effect on {d['outcome'].iloc[0]}")
        ax.legend(title="MR Test", loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        ax.set_title(f"{id_exp} → {id_out}")
        ax.grid(True)

        plots[(id_exp, id_out)] = fig

    return plots


def blank_plot(message):
    fig, ax = plt.subplots()
    ax.text(0, 0, message, ha='center', va='center', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_frame_on(False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return fig