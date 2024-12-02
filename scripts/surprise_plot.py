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

from hydrodiy.plot import putils, violinplot, boxplot

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

    putils.line(ax, 1, 0, 0, 0, "k-", lw=0.7)

    # Hide the right and top spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


def set_logscale(ax):
    y0, y1 = ax.get_ylim()
    y0 = int(min(y0, y1))-1
    y1 = int(max(y0, y1))+2
    yticks = np.arange(y0, y1+1)
    ax.set_yticks(yticks)
    yticklabs = [f"{10.**t:0.0f}" if t>=0 else f"{10.**t:0.1f}" \
                        for t in yticks]
    ax.set_yticklabels(yticklabs)


def main():
    #----------------------------------------------------------------------
    # Config
    #----------------------------------------------------------------------
    varnames = [
        "SPECIFICFLOW_PEAK", \
        "RUNOFF_120H", \
        "RUNOFF_240H"
    ]

    censored = False

    cn_surprise = "Q100-SURPRISE"

    marginals = ["GEV", "LOGPEARSON3"]

    axwidth = 18
    axheight = 4.5
    fdpi = 300
    xlabel_fontsize = 14
    ylabel_fontsize = 14

    color_22 = "tab:red"
    color_others = "tab:purple"
    fname_22 = "Feb22"

    title_args = dict(y=0.98, x=0.01, pad=-15, loc="left", \
                        fontweight="normal", fontsize=20)

    #------------------------------------------------------------
    # Folders
    #------------------------------------------------------------
    source_file = Path(__file__).resolve()

    froot = source_file.parent.parent

    fsrc = froot / "data"

    fimg = froot / "images" / "surprise"
    fimg.mkdir(exist_ok=True, parents=True)

    #------------------------------------------------------------
    # Get data
    #------------------------------------------------------------

    # Site info
    fs = fsrc / "sites_info.csv"
    sites = pd.read_csv(fs, index_col="STATIONID", skiprows=8)

    # Flood event data
    fe = fsrc / "flood_data_censored.zip" if censored else fsrc / "flood_data.zip"
    eventdata = pd.read_csv(fe, dtype={"siteid": str}, skiprows=8)

    # Major australian floods
    fm = fsrc / "major_floods.csv"
    major_floods = pd.read_csv(fm, index_col="FLOODID", \
                            parse_dates=["START_DATE", "END_DATE"], \
                            skiprows=8)
    idx = major_floods.MORE_THAN_5_SITES_AVAILABLE==1
    major_floods = major_floods.loc[idx]

    #------------------------------------------------------------
    # Plot
    #------------------------------------------------------------
    for marginal in marginals:
        print(f"Plot {marginal}")

        # Export data
        pat = "SITEID|STATE|MAJOR_FLOOD|WATERYEAR"

        vns = [re.sub("SPECIFIC", "", re.sub("RUNOFF", "VOL", vn)) for vn in varnames]
        pat += "|"+"|".join([f"{vn}_{marginal}-{cn_surprise}" for vn in vns])
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

            fc = fe.parent / f"{fe.stem}_counts_{cn}.csv"
            counts.to_csv(fc)

        # Fig dimensions
        mosaic = [[f"{va}/ts"] for va in varnames]
        ncols, nrows = len(mosaic[0]), len(mosaic)

        plt.close("all")
        nrows, ncols = len(mosaic), len(mosaic[0])
        fig = plt.figure(figsize=(axwidth*ncols, axheight*nrows), \
                                                    layout="constrained")

        wr = [2, 1] if ncols==2 else [1]
        kw = dict(width_ratios=wr, hspace=0.1)
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
        highlighted_any = set()

        # Plot
        for iax, (aname, ax) in enumerate(axs.items()):
            varname, _ = aname.split("/")

            # Get surprise data
            col_value = next(cn for cn in eventdata.columns \
                                        if re.search(varname, cn))
            mfdata = pd.pivot_table(eventdata, index="SITEID", \
                                    columns="MAJOR_FLOOD", \
                                    values=col_value)

            # .. re-order floods
            for cn in major_floods.index:
                if not cn in mfdata.columns:
                    mfdata.loc[:, cn] = np.nan

            mfdata = mfdata.loc[:, major_floods.index]
            mfdata.columns = major_floods.SHORTNAME

            # box plot
            bp = boxplot.Boxplot(mfdata, \
                            style="narrow", \
                            show_median=False,\
                            show_text=False)
            bp.draw(ax=ax)

            # color NR 22
            st = bp.stats
            colms = st.columns
            highlighted = st.loc["mean", colms!=fname_22].nlargest(4).index.tolist()
            highlighted += [fname_22]

            highlighted_any.update(set(highlighted))

            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            for fn in highlighted:
                col = color_22 if fn == fname_22 else color_others
                bp.set_color(fn, col, alpha=0.9)

                ct = bp.stats.loc["count", fn]
                m = bp.stats.loc["mean", fn]+(y1-y0)/50
                x = colms.get_loc(fn)
                ax.text(x, m, f"{m:0.2f}", color=col, fontweight="bold", \
                                va="bottom", ha="center", fontsize=14, \
                                path_effects=[pe.withStroke(linewidth=6, \
                                                    foreground="w")])

                if fn == fname_22:
                    v = bp.stats.loc[["25.0%", "75.0%"], fn].values
                    ll = (x0, v[0])
                    width  = x1-x0
                    height = v[1]-v[0]
                    rect = Rectangle(ll, width, height, \
                                        facecolor=col, alpha=0.2)
                    ax.add_patch(rect)

                    m = bp.stats.loc["mean", fn]
                    putils.line(ax, 1, 0, 0, m, lw=0.5, color=col)

            y0 = max(-0.2, y0)
            #y1 = max(y1, topn.max()*1.1)
            ax.set_ylim((y0, y1))

            # decorate
            ax.set_xticks(np.arange(mfdata.shape[1]))
            if aname==mosaic[-1][0]:
                ax.set_xticklabels(mfdata.columns)
                ax.tick_params(axis="x", labelrotation=90)
            else:
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.spines.bottom.set_visible(False)

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
                vn = "Specific peak flow"

            ylabel = "Surprise index [-]"
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
            con = ConnectionPatch(\
                        xyA=(xf, y1), coordsA=ax1.transData, \
                        xyB=(xf, y2), coordsB=ax2.transData, \
                        color="0.6", linestyle=":")
            fig.add_artist(con)

        fp = fimg / (f"surprise"+\
                f"_{marginal}_C{censored}.png")
        fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    main()
