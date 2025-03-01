#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-03-22 Fri 04:19 PM
## Comment : Plot flood event summary and surprise
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path
from itertools import product as prod

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

from dateutil.relativedelta import relativedelta as delta
from string import ascii_lowercase as letters
from calendar import month_abbr as months

import numpy as np
import pandas as pd

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as pe
from matplotlib import ticker

def plot_line(ax, slope, x0, y0, *args, **kwargs):
    xa, xb = ax.get_xlim()
    ylim = ax.get_ylim()
    ya = slope*(xa-x0)+y0
    yb = slope*(xb-x0)+y0
    ax.plot([xa, xb], [ya, yb], *args, **kwargs)

    ax.set_xlim((xa, xb))
    ax.set_ylim(ylim)


def format_spines(ax):
    xtks = ax.get_xticks()
    ytks = ax.get_yticks()

    # Only draw spines for the data range, not in the margins
    xb0, xb1 = ax.get_xbound()
    if len(xtks)>0:
        xtks = xtks[(xtks>=xb0)&(xtks<=xb1)]
        x0 = xtks.min()
        x1 = xtks.max()
        ax.set_xticks(xtks)
        ax.spines.bottom.set_bounds(x0, x1)

    yb0, yb1 = ax.get_ybound()
    ytks = ytks[(ytks>=yb0)&(ytks<=yb1)]
    if len(ytks)>4:
        ytks = ytks[::2]
    y0 = ytks.min()
    y1 = ytks.max()
    ax.set_yticks(ytks)
    ax.spines.left.set_bounds(y0, y1)

    plot_line(ax, 0, 0, 0, "k-", lw=0.7)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


def set_logscale(ax):
    y0, y1 = ax.get_ylim()
    y0 = int(min(y0, y1)) - 1
    y1 = int(max(y0, y1)) + 2
    yticks = np.arange(y0, y1 + 1)
    ax.set_yticks(yticks)
    yticklabs = [f"{10.**t:0.0f}" if t >= 0 else f"{10.**t:0.1f}"
                 for t in yticks]
    ax.set_yticklabels(yticklabs)


def main(censored, marginal, hide_points, clear=False):
    #----------------------------------------------------------------------
    # Config
    #----------------------------------------------------------------------
    print(f"Censored = {censored}")
    print(f"Marginal = {marginal}")
    print(f"Hide points = {hide_points}\n")

    varnames = ["SPECIFICFLOW_PEAK",
                "RUNOFF_240H"
                ]

    cn_surprise = "Q100-SURPRISE"

    axwidth = 18
    axheight = 7
    fdpi = 300
    xlabel_fontsize = 14
    ylabel_fontsize = 14

    color_22 = "tab:red"
    color_others = "tab:purple"
    fname_22 = "Feb22"

    title_args = dict(y=0.98, x=0.01, pad=-15, loc="left",
                      fontweight="normal", fontsize=20)

    #------------------------------------------------------------
    # Folders
    #------------------------------------------------------------
    source_file = Path(__file__).resolve()

    froot = source_file.parent.parent

    fsrc = froot / "data"

    fimg = froot / "images" / "surprise"
    fimg.mkdir(exist_ok=True, parents=True)

    if clear:
        for f in fimg.glob("*.*"):
            f.unlink()

    #------------------------------------------------------------
    # Get data
    #------------------------------------------------------------

    # Site info
    fs = fsrc / "sites_info.csv"
    sites = pd.read_csv(fs, index_col="STATIONID", skiprows=9)

    # Flood event data
    fe = fsrc / "floods" / "flood_data_censored.zip" if censored \
        else fsrc / "floods" / "flood_data.zip"
    eventdata = pd.read_csv(fe, dtype={"siteid": str}, skiprows=9)

    # Major australian floods
    fm = fsrc / "floods" / "major_floods.csv"
    major_floods = pd.read_csv(fm, index_col="FLOODID",
                               parse_dates=["START_DATE", "END_DATE"],
                               skiprows=9)
    idx = major_floods.MORE_THAN_5_SITES_AVAILABLE==1
    major_floods = major_floods.loc[idx]

    #------------------------------------------------------------
    # Plot
    #------------------------------------------------------------
    # Select data
    pat = "SITEID|STATE|MAJOR_FLOOD|WATERYEAR"
    vns = [re.sub("SPECIFIC", "", re.sub("RUNOFF", "VOL", vn)) for vn in varnames]
    pcens = 0.2 if censored else 0.
    pat += "|"+"|".join([f"{vn}_C{pcens*10:02.0f}_{marginal}-{cn_surprise}" for vn in vns])
    edata = eventdata.filter(regex=pat, axis=1)

    idx = eventdata.filter(regex="SPECIFICFLOW", axis=1).squeeze().notnull()
    idx &= eventdata.MAJOR_FLOOD.notnull()
    edata = edata.loc[idx, :]

    for cn in ["SITEID", "MAJOR_FLOOD"]:
        counts = edata.groupby(cn).apply(lambda x: x.notnull().sum())
        counts.loc[:, "diff"] = counts.iloc[:, 0]-counts.iloc[:, 1]
        if cn == "MAJOR_FLOOD":
            idx = counts.index[counts.index.isin(major_floods.index)]
            st = major_floods.START_DATE[idx].astype(np.int64)
            en = major_floods.START_DATE[idx].astype(np.int64)
            dt = pd.to_datetime((st+en)/2)
            counts.loc[:, "DATE"] = pd.NaT
            counts.loc[idx, "DATE"] = dt.values
            counts = counts.sort_values("DATE")
        else:
            counts = counts.sort_values("diff", ascending=False)

        fc = fimg / f"{fe.stem}_C{censored}_counts_{cn}.csv"
        counts.to_csv(fc)

    # Fig dimensions
    mosaic = [[vn] for vn in varnames]
    ncols, nrows = len(mosaic[0]), len(mosaic)

    plt.close("all")
    nrows, ncols = len(mosaic), len(mosaic[0])
    fig = plt.figure(figsize=(axwidth*ncols, axheight*nrows),
                     layout="constrained")

    wr = [2, 1] if ncols == 2 else [1]
    kw = dict(width_ratios=wr, hspace=0.1)
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
    highlighted_any = set()

    # Plot
    for iax, (varname, ax) in enumerate(axs.items()):
        # Get surprise data
        vn = re.sub("SPECIFIC", "", varname)
        vn = re.sub("RUNOFF", "VOL", vn)
        col_value = next(cn for cn in edata.columns \
            if re.search(vn, cn))
        mfdata = pd.pivot_table(edata, index="SITEID",
                                columns="MAJOR_FLOOD",
                                values=col_value)

        # .. re-order floods
        for cn in major_floods.index:
            if not cn in mfdata.columns:
                mfdata.loc[:, cn] = np.nan

        mfdata = mfdata.loc[:, major_floods.index]
        mfdata.columns = [re.sub("-.*", "", n) \
                            for n in major_floods.SHORTNAME]

        # Highlighted floods
        means = mfdata.mean()
        highlighted = means[means.index!=fname_22].nlargest(4).index.tolist()
        highlighted += [fname_22]
        highlighted_any.update(set(highlighted))

        # Boxplot
        xtk = np.arange(mfdata.shape[1])
        xlim = xtk[0]-0.5, xtk[-1]+0.5

        for ifn, (fn, se) in enumerate(mfdata.items()):
            # .. colors
            col = "tab:blue"
            if fn in highlighted:
                col = color_22 if fn==fname_22 else color_others

            wl, q1, med, q3, wh = se.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            mean = se.mean()

            # .. Feb22 highlight
            if fn == fname_22:
                rec = Rectangle((xlim[0], q1), xlim[1]-xlim[0], q3-q1, \
                                    facecolor=col, alpha=0.2)
                ax.add_patch(rec)
                ax.plot(xlim, [mean, mean], color=col, lw=1)

            # .. boxes
            x = xtk[ifn]
            dx = 0.9
            rec = Rectangle((x-dx/2, q1), dx, q3-q1,\
                                facecolor=col, alpha=1.0)
            ax.add_patch(rec)

            # .. points
            dx = 0.2
            if not hide_points:
                xx = x+dx/2*np.random.uniform(-1, 1, len(se))
                ax.plot(xx, se, "o", ms=6, mfc=col, mec="w", alpha=0.5)

            # .. means
            ax.plot(x, mean, "o", mec=col, ms=8, mfc="w", lw=2)
            if fn in highlighted:
                dy = 0.02
                ax.text(x, mean+dy, f"{mean:0.2f}", color=col, fontweight="bold",
                        va="bottom", ha="center", fontsize=15,
                        path_effects=[pe.withStroke(linewidth=7,
                                                    foreground="w")])
        y0, y1 = (-0.1, 0.4)
        ax.set_ylim((y0, y1))

        # decorate
        if varname==mosaic[-1][0]:
            ax.set_xticks(xtk)
            ax.set_xticklabels(mfdata.columns.to_list())
            ax.tick_params(axis="x", labelrotation=90)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.spines.bottom.set_visible(False)

        ax.set_xlim(xlim)

        # Format spines
        format_spines(ax)

        # Set labels
        vartxt = varname.lower()
        if vartxt.startswith("rain_"):
            n = int(re.search("[0-9]+", varname).group())/24
            vn = f"Rainfall total - {n:0.0f} days"
        elif vartxt.startswith("runoff_"):
            n = int(re.search("[0-9]+", varname).group())/24
            vn = f"Runoff total - {n:0.0f} days"
        elif vartxt.startswith("specificflow_peak"):
            vn = "Specific instantaneous peak flow"

        ylabel = "1% AEP Surprise index [-]"
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        title_txt = vn

        title = f" ({letters[iax]}) {title_txt}"
        ax.set_title(title, **title_args)

    # Set font of x label
    ax = axs[mosaic[-1][0]]
    for xlab in ax.get_xticklabels():
        xlab.set_fontsize(xlabel_fontsize)
        fn = xlab.get_text()
        if fn in highlighted_any:
            col = color_22 if fn == fname_22 else color_others
            xlab.set_color(col)
            xlab.set_fontweight("bold")
            xlab.set_fontsize(15)

    # Highlight floods
    ax1 = axs[mosaic[0][0]]
    _, y1 = ax1.get_ylim()

    ax2 = axs[mosaic[-1][0]]
    cols = mfdata.columns.tolist()
    y2, _ = ax2.get_ylim()

    for fname in mfdata.columns[::3]:
        xf = cols.index(fname)
        con = ConnectionPatch(xyA=(xf, y1),
                              coordsA=ax1.transData,
                              xyB=(xf, y2),
                              coordsB=ax2.transData,
                              color="0.6", linestyle=":")
        fig.add_artist(con)

    if varname == varnames[-1]:
        ax.set_xlabel("Regional flood event", fontsize=20)

    fp = fimg / (f"surprise"
                 + f"_{marginal}_C{int(censored)}_H{int(hide_points)}.png")
    fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(\
        description="Surprise index figure", \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-hp", "--hide_points", help="Show individual points",
                        action="store_true", default=False)
    args = parser.parse_args()
    hide_points = args.hide_points

    # Run script for both uncensored and censored fit
    main(False, "GEV", hide_points, True)
    main(False, "LOGPEARSON3", hide_points)
    main(True, "GEV", hide_points)
    main(True, "LOGPEARSON3", hide_points)
