#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-03-22 Fri 03:25 PM
## Comment : Plot flood stats as scatter plots
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path
from string import ascii_letters as letters
from itertools import product as prod

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib as mpl

# Select backend
#mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def kde(x, y, extent, ngrid=100, levels=[0.9, 0.95, 0.99]):
    # Fit kernel
    values = np.column_stack([x, y]).T
    kernel = gaussian_kde(values)

    # Create interpolation grid
    xx = np.linspace(extent[0], extent[1], ngrid)
    yy = np.linspace(extent[2], extent[3], ngrid)
    X, Y = np.meshgrid(xx, yy)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel.evaluate(positions).T, X.shape)

    # Defines levels as given probability mass
    # above threshold
    threshs = np.linspace(Z.min(), Z.max(), 1000)
    integ = np.zeros_like(threshs)
    V = Z / Z.sum()
    for i, t in enumerate(threshs):
        integ[i] = V[Z > t].sum()

    pdf_threshs = pd.Series(np.interp(levels, integ[::-1],
                                      threshs[::-1]),
                            index=levels).sort_values()
    return X, Y, Z, pdf_threshs, kernel


def main(version):
    #----------------------------------------------------------------------
    # Config
    #----------------------------------------------------------------------
    title1 = "Specific Instantaneous Peak Flow vs. Catchment Area"
    title2 = "Specific Instantaneous Peak Flow vs. Ten Days Runoff Total"
    plots = {
        title1: dict(varx="CATCHMENTAREA",
                  vary="SPECIFICFLOW_PEAK"),
        title2: dict(varx="RUNOFF_240H",
                  vary="SPECIFICFLOW_PEAK"),
    }

    var_axislabels = {
        "SPECIFICFLOW_PEAK": "Specific peak flow [m$^3$ s$^{-1}$ km$^{-2}$]",
        "CATCHMENTAREA": "Catchment Area [km$^2$]",
        "RUNOFF_120H": "Five days runoff Total [mm]",
        "RUNOFF_240H": "Ten days runoff Total [mm]"
        }

    imgext = "png"
    axwidth, axheight = 6, 6
    fdpi = 300

    #----------------------------------------------------------------------
    # Folders
    #----------------------------------------------------------------------
    source_file = Path(__file__).resolve()

    froot = source_file.parent.parent

    fsrc  = froot / "data"

    fimg = froot / "images" / "scatter"
    fimg.mkdir(exist_ok=True, parents=True)
    for f in fimg.glob("*.png"):
        f.unlink()

    #------------------------------------------------------------
    # Get data
    #------------------------------------------------------------

    # Flood event data
    fe = fsrc / "floods" / f"flood_data_v{version}.zip"
    eventdata = pd.read_csv(fe, dtype={"SITEID": str},
                            comment="#")

    # Major australian floods
    fm = fsrc / "floods" / "major_floods.csv"
    major_floods = pd.read_csv(fm, index_col="FLOODID",
                               parse_dates=["START_DATE", "END_DATE"],
                               comment="#")
    major_floods = major_floods.sort_values("START_DATE")

    # .. set major floods plot specs
    inr = major_floods.index=="NorthernRivers-Feb22"
    ideb = major_floods.index=="CycDebbie-Mar17"
    i74 = major_floods.index=="EastAustralia-Feb74"

    major_floods.loc[:, "marker"] = ""
    major_floods.loc[inr, "marker"] = "o"
    major_floods.loc[ideb, "marker"] = "x"
    major_floods.loc[i74, "marker"] = "+"

    major_floods.loc[:, "color"] = ""
    major_floods.loc[inr, "color"] = "tab:red"
    major_floods.loc[ideb, "color"] = "tab:green"
    major_floods.loc[i74, "color"] = "tab:purple"

    idx = major_floods.MORE_THAN_5_SITES_AVAILABLE==1
    major_floods = major_floods.loc[idx]

    #------------------------------------------------------------
    # Plot
    #------------------------------------------------------------

    plt.close("all")
    mosaic = [[p for p in plots]]
    nrows, ncols = len(mosaic), len(mosaic[0])
    fig = plt.figure(figsize=(axwidth*ncols, axheight*nrows),
                     layout="tight")
    axs = fig.subplot_mosaic(mosaic)
    facts = {}
    for iax, (title, ax) in enumerate(axs.items()):
        varx = plots[title]["varx"]
        vary = plots[title]["vary"]

        print(f"Plot {iax + 1} : {varx}/{vary}")
        pat = f"SITEID|MAJOR_FLOOD|STATE|{vary}\\[|{varx}\\["
        df = eventdata.filter(regex=pat, axis=1)

        # .. explanatory variable
        cn = next(c for c in df.columns if re.search(varx, c))
        x = df.loc[:, cn]

        # .. target variable
        cn = next(c for c in df.columns if re.search(vary, c))
        y = df.loc[:, cn]
        nvalid = y.notnull().sum()

        # plot
        nev = len(x)
        ns = df.SITEID.unique().shape[0]
        label = f"Site event"
        alpha = 0.1
        ax.plot(x, y, "o", alpha=alpha, mec="none", mfc="0.6", ms=3)
        ax.plot([], [], "o", mec="k", mfc="0.8", ms=5,
                label=label)

        # Plot historical floods
        # Define 2022 event
        for mfid, mfinfo in major_floods.iterrows():
            iflood = df.MAJOR_FLOOD == mfid
            xf, yf = x[iflood], y[iflood]
            insw = iflood & (df.STATE=="NSW")

            # Show site id for NR 2022
            mfids = ["NorthernRivers-Feb22"]
            if not mfid in mfids:
                continue

            # Plot NR 2022
            n = len(xf)
            lab = f"{re.sub('.*-', '', mfid)} regional event"
            ax.plot(xf, yf, mfinfo.marker, color=mfinfo.color,
                    mec="0.3", label=lab)

        # Axis scales
        if varx == "CATCHMENTAREA":
            xlim = 6, 1e5
            ax.set(xscale="log", xlim=xlim)
        elif varx.startswith("RUNOFF"):
            xlim = 10, ax.get_xlim()[1]
            ax.set(xscale="log", xlim=xlim)

        if vary.startswith("SPECIFIC"):
            ylim = 1e-1, 8e1
            ax.set(yscale="log", ylim=ylim)
        elif vary.startswith("RUNOFF"):
            ylim = 10, ax.get_ylim()[1]
            ax.set(yscale="log", ylim=ylim)

        if title == title1:
            # Max envelop
            xmax, ymax = [], []
            nbnds = 20
            bnds = np.logspace(math.log10(1e-4+x.min()), math.log10(x.max()), nbnds)
            for ibnd in range(1, len(bnds)-1):
                b0, b1 = bnds[ibnd:ibnd+2]
                kk = (x>=b0) & (x<b1)
                if kk.sum()>0:
                    xmax.append((b0+b1)/2)
                    ymax.append(y[kk].max())

            ax.plot(xmax, ymax, ":", color="grey", lw=1.5,
                    label="Max AUS")

            # Quantile regression
            qtle = 0.99
            xx = np.log(x) if ax.get_xscale()=="log" else x
            yy = np.log(y) if ax.get_yscale()=="log" else y
            data = pd.concat([xx, yy], axis=1)
            data.columns = [re.sub("\[.*|-./*", "", cn) for cn in data.columns]
            iok = (np.isfinite(data)&data.notnull()).all(axis=1)
            data = data.loc[iok]
            mod = smf.quantreg(f"{data.columns[1]}~{data.columns[0]}", data)
            res = mod.fit(q=qtle)
            a, b = res.params.Intercept, res.params[data.columns[0]]

            x0, x1 = ax.get_xlim()
            if ax.get_xscale()=="log":
                uu = np.logspace(math.log10(1-3+x0), math.log10(x1), 500)
                vv = a+b*np.log(uu)
                eq = f"${a:0.1f}\\ A^{{{b:0.2f}}}$"
            else:
                uu = np.linspace(x0, x1, 500)
                vv = a+b*uu
                eq = f"${a:0.1f}+{b:0.2f}\\times X$"

            vv = np.exp(vv) if ax.get_yscale()=="log" else vv

            ax.plot(uu, vv, "k-", label=f"99% AUS ({eq})", lw=3)
            ax.set_xlim((x0, x1))

            i2022 = df.MAJOR_FLOOD == "NorthernRivers-Feb22"
            above_au = np.log(y[i2022]) - a - b * np.log(x[i2022])
            above_au = int((above_au >= 0).sum())

            # Reference curves
            x0, x1 = ax.get_xlim()
            xx = np.logspace(math.log10(x0), math.log10(x1), 500)

            # See Table 1 in
            # O’Connor, Jim E., and John E. Costa. ‘Spatial Distribution
            # of the Largest Rainfall-Runoff Floods from Basins between 2.6
            # and 26,000 Km2 in the United States and Puerto Rico’. Water
            # Resources Research 40, no. 1 (2004). https://doi.org/10.1029/2003WR002247.
            # Adapted from Table 1
            yy = 74*xx**(0.53-1)
            eq = r"$74.0\ A^{-0.47}$"
            ax.plot(xx, yy, "--", color="tab:purple", label=f"99% US ({eq})", lw=3)
            ax.set_xlim((x0, x1))

            above_us = np.log(y[i2022]) - math.log(74) + 0.47 * np.log(x[i2022])
            above_us = int((above_us >= 0).sum())

        else:
            # Contour plot
            idx = ~np.isnan(x) & ~np.isnan(y)
            xlog, ylog = np.log10(x[idx]), np.log10(y[idx])
            extent = np.log10(np.array([xlim[0], xlim[1], ylim[0], ylim[1]]))

            Xlog, Ylog, Z, levels, kernel = kde(xlog, ylog, extent, ngrid=100)

            i2022 = df.MAJOR_FLOOD.loc[idx] == "NorthernRivers-Feb22"
            obs = np.vstack([xlog[i2022], ylog[i2022]])
            pobs = kernel.evaluate(obs)
            between_95_99 = (pobs >= levels.loc[0.99]) & (pobs < levels.loc[0.95])
            between_95_99 = int(between_95_99.sum())
            outside_of_99 = int((pobs < levels.loc[0.99]).sum())

            axi = ax.inset_axes([0, 0, 1, 1])
            level_colors=["k", "blueviolet", "violet"]
            CS = axi.contour(Xlog, Ylog, Z, levels=levels.values,
                             colors=level_colors, lw=2)

            # .. modify contour labels
            fmt = {}
            for l in CS.levels:
                fmt[l] = f"{100 * levels.index[levels == l][0]:0.0f}%"

            axi.clabel(CS, CS.levels, fmt=fmt, colors=level_colors)

            # .. set legend item
            for i, collec in enumerate(CS.collections):
                lab = f"KDE probability mass {100 * levels.index[i]:0.0f}%"
                col = collec.get_edgecolor()
                lw = collec.get_linewidth()
                ax.plot([], [], "-", color=col, lw=lw, label=lab)

            axi.axis("off")

        # decorate
        xlabel = var_axislabels[varx]
        ylabel = var_axislabels[vary]
        title_full = f"({letters[iax]}) {title}"
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title_full)
        legloc = 1 if title == title1 else 2
        ax.legend(loc=legloc, fontsize="medium", framealpha=0.0)

        facts[title] = {
            "number_of_sites": len(df.SITEID.unique()),
            "number_of_events": len(x)
            }

        if title == title1:
            facts[title]["2022_above_99AU"] = above_au
            facts[title]["2022_above_99US"] = above_us
        elif title == title2:
            facts[title]["2022_between_95_99"] = between_95_99
            facts[title]["2022_outside_of_99"] = outside_of_99

    fp = fimg / f"FIGB_scatterplots_v{version}.{imgext}"
    fig.savefig(fp, dpi=fdpi)

    ff = fp.parent / f"{fp.stem}_facts_v{version}.json"
    with ff.open("w") as fo:
        json.dump(facts, fo, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scatter plot of site events characteristics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="Version number",
                        type=str, default="png")
    args = parser.parse_args()
    version = args.version

    main(version)
