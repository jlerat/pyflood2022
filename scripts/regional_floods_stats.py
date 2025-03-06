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

from string import ascii_lowercase as letters

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib.patches import ConnectionPatch, Rectangle

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt

from surprise_plot import format_spines


def main(clear=False):
    #----------------------------------------------------------------------
    # Config
    #----------------------------------------------------------------------
    censored = False
    marginal = "GEV"
    cn_surprise = "Q100-SURPRISE"

    mosaic = [["COUNT"] * 2, ["COUNT_HIST", "MONTH_HIST"]]

    color_22 = "tab:red"
    fname_22 = "Feb22"
    facecolor_default = "lightblue"
    edgecolor_default = "navy"
    axwidth = 10
    axheight = 7
    fdpi = 300
    ticklabel_fontsize = 13
    axlabel_fontsize = 17

    title_args = dict(y=0.98, x=0.01, loc="left",
                      fontweight="normal", fontsize=20)

    #------------------------------------------------------------
    # Folders
    #------------------------------------------------------------
    source_file = Path(__file__).resolve()

    froot = source_file.parent.parent

    fsrc = froot / "data"

    fimg = froot / "images" / "regional_floods_stats"
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
    eventdata = pd.read_csv(fe, dtype={"siteid": str},
                            parse_dates=["FLOW_TIME_OF_PEAK"], skiprows=9)

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
    pat = "SITEID|MAJOR_FLOOD|FLOW_TIME_OF_PEAK"
    edata = eventdata.filter(regex=pat, axis=1)
    edata = edata.loc[edata.MAJOR_FLOOD.notnull()]
    t = pd.DatetimeIndex(edata.FLOW_TIME_OF_PEAK)
    edata.loc[:, "FLOW_MONTH_OF_PEAK"] = t.month

    col_value = "FLOW_MONTH_OF_PEAK"
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

    # Fig dimensions
    ncols, nrows = len(mosaic[0]), len(mosaic)

    plt.close("all")
    nrows, ncols = len(mosaic), len(mosaic[0])
    fig = plt.figure(figsize=(axwidth*ncols, axheight*nrows),
                     layout="constrained")
    kw = dict(hspace=0.1, wspace=0.1)
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    # Plot
    for iax, (varname, ax) in enumerate(axs.items()):
        if varname.endswith("HIST"):
            if varname.startswith("MONTH"):
                data = mfdata.apply(lambda x: x.value_counts()).idxmax()
                data = data.value_counts()
                for m in range(1, 13):
                    if m not in data.index:
                        data.loc[m] = 0
                data = data.sort_index()
                data.index = data.index.astype(int)

                title_txt = "In which months do regional events occur?"

            elif varname.startswith("COUNT"):
                data = mfdata.notnull().sum()
                data = pd.cut(data, np.arange(9)*50).value_counts()\
                    .sort_index()
                data.index = [f"{i.left}\nto {i.right}" for i in data.index]

                title_txt = "How many site events are selected\nfor one regional event?"

            data.plot(ax=ax, kind="bar", rot=0)
        else:
            # Boxplot
            xtk = np.arange(mfdata.shape[1])
            xlim = xtk[0]-0.5, xtk[-1]+0.5

            for ifn, (fn, se) in enumerate(mfdata.items()):
                # .. colors
                fcol = color_22 if fn == fname_22 else facecolor_default
                ecol = color_22 if fn == fname_22 else edgecolor_default

                count = se.notnull().sum()
                x = xtk[ifn]
                dx = 0.9
                rec = Rectangle((x-dx/2, 0), dx, count,
                                facecolor=fcol, alpha=1.0,
                                edgecolor=ecol)
                ax.add_patch(rec)
                ax.add_patch(rec)
                y0, y1 = (0., 400)
                if fn == fname_22:
                    ax.plot(xlim, [count, count], color=fcol, lw=1)

            y0, y1 = (0., 400.)
            ax.set_ylim((y0, y1))

            title_txt = "Number of site events for each regional event"

        # decorate
        if varname==mosaic[0][0]:
            ax.set_xticks(xtk)
            ax.set_xticklabels(mfdata.columns.to_list())
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_xlim(xlim)

        ax.tick_params(axis='both', which='major',
                       labelsize=ticklabel_fontsize)
        ax.tick_params(axis='both', which='minor',
                       labelsize=ticklabel_fontsize)

        # Format spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Set labels
        ylabel = "Number of regional event [-]" if varname.endswith("HIST") \
            else "Number of site events [-]"
        ax.set_ylabel(ylabel, fontsize=axlabel_fontsize)

        title = f" ({letters[iax]}) {title_txt}"
        ax.set_title(title, **title_args)

        # Set font of x label
        if varname == "COUNT":
            ax = axs[mosaic[0][0]]
            for xlab in ax.get_xticklabels():
                xlab.set_fontsize(ticklabel_fontsize)
                fn = xlab.get_text()
                if fn == fname_22:
                    col = color_22
                    xlab.set_color(col)
                    xlab.set_fontweight("bold")
                    xlab.set_fontsize(ticklabel_fontsize)
            xlabel = "Regional event"
        elif varname == "COUNT_HIST":
            xlabel = "Number of site events per regional event [-]"
        else:
            xlabel = "Calendar Month of regional event [-]"

        ax.set_xlabel(xlabel, fontsize=axlabel_fontsize)

    fp = fimg / "flood_stats.png"
    fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    main(True)
