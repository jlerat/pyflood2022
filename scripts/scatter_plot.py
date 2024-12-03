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
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def main():
    #----------------------------------------------------------------------
    # Config
    #----------------------------------------------------------------------
    varnames = [
        "SPECIFICFLOW_PEAK", \
        "RUNOFF_120H", \
        "RUNOFF_240H"
    ]

    # used for XY config
    varref = "SPECIFICFLOW_PEAK"

    varexplain = "CATCHMENTAREA"

    imgext = "png"
    axwidth, axheight = 5, 5
    fdpi = 300

    ncols = min(3, len(varnames))

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
    fe = fsrc / "flood_data.zip"
    eventdata = pd.read_csv(fe, dtype={"SITEID": str}, skiprows=8)

    # Major australian floods
    fm = fsrc / "major_floods.csv"
    major_floods = pd.read_csv(fm, index_col="FLOODID", \
                            parse_dates=["START_DATE", "END_DATE"], \
                            skiprows=8)
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

    # Fig dimensions
    nvars = len(varnames)
    offset = 0 if nvars%ncols==0 else 1
    nrows = nvars//ncols+offset

    vv = varnames
    mosaic = [l.tolist()+["."]*(ncols-len(l))\
                    for l in np.array_split(vv, nrows)]

    flat = [n for m in mosaic for n in m if n!="."]
    extremes = []

    # plot
    plt.close("all")
    nrows, ncols = len(mosaic), len(mosaic[0])
    fig = plt.figure(figsize=(axwidth*ncols, axheight*nrows), \
                                                layout="tight")
    axs = fig.subplot_mosaic(mosaic)

    for iax, (varname, ax) in enumerate(axs.items()):
        vartxt = varname.lower()

        print(f"Plot {varname}")
        pat = f"SITEID|MAJOR_FLOOD|STATE|{varname}\\[|{varexplain}\\["
        df = eventdata.filter(regex=pat, axis=1)

        # .. explanatory variable
        cn = next(c for c in df.columns if re.search(varexplain, c))
        x = df.loc[:, cn]

        # .. target variable
        cn = next(c for c in df.columns if re.search(varname, c))
        y = df.loc[:, cn]
        nvalid = y.notnull().sum()

        # plot
        alpha = 0.1
        ax.plot(x, y, "o", alpha=alpha, mec="none", mfc="0.6", ms=3)

        # Max envelop
        xmax, ymax = [], []
        nbnds = 15
        bnds = np.logspace(math.log10(1e-4+x.min()), math.log10(x.max()), nbnds)
        for ibnd in range(1, len(bnds)-1):
            b0, b1 = bnds[ibnd:ibnd+2]
            kk = (x>=b0) & (x<b1)
            if kk.sum()>0:
                xmax.append((b0+b1)/2)
                ymax.append(y[kk].max())

        ax.plot(xmax, ymax, ":", color="grey", lw=2, \
                label="Max AUS")

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
            lab = re.sub(".*-", "", mfid)
            ax.plot(xf, yf, mfinfo.marker, color=mfinfo.color, \
                                mec="0.3", label=lab)

        # decorate
        ax.set(xscale="log", xlim=(6, 1e5))
        xlabel = "Dist Coast [km]" if varexplain == "coastdist" \
                    else "Catchment Area [km$^2$]"

        unit = r"m$^3$ s$^{-1}$ km$^{-2}$" if varname.startswith("SPECIFIC") else "mm"

        vartxt = varname.lower()
        if re.search("specific", vartxt):
            vtitle = "Specific flow"
        elif re.search("^(rain|runoff)", vartxt):
            vtitle = re.sub("_.*", "", vartxt).title()

        ylabel = f"{vtitle} [{unit}]"

        logpat = "runoff_|specific"
        if re.search(logpat, varname.lower()):
            ax.set(yscale="log")

        ax.set(xlabel=xlabel, ylabel=ylabel)

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
            eq = f"${a:0.1f}\\times A^{{{b:0.2f}}}$"
        else:
            uu = np.linspace(x0, x1, 500)
            vv = a+b*uu
            eq = f"${a:0.1f}+{b:0.2f}\\times X$"

        vv = np.exp(vv) if ax.get_yscale()=="log" else vv

        ax.plot(uu, vv, "k-", label=f"99% AUS ({eq})", lw=3)
        ax.set_xlim((x0, x1))

        # Reference envelop curves
        x0, x1 = ax.get_xlim()
        if re.search("specificflow_peak", vartxt) \
                            and varexplain=="CATCHMENTAREA":
            xx = np.logspace(math.log10(x0), math.log10(x1), 500)

            # See Table 1 in
            # O’Connor, Jim E., and John E. Costa. ‘Spatial Distribution
            # of the Largest Rainfall-Runoff Floods from Basins between 2.6
            # and 26,000 Km2 in the United States and Puerto Rico’. Water
            # Resources Research 40, no. 1 (2004). https://doi.org/10.1029/2003WR002247.
            # Adapted from Table 1
            yy = 74*xx**(0.53-1)
            eq = r"$74\times A^{-0.47}$"
            ax.plot(xx, yy, "--", color="tab:purple", label=f"99% US ({eq})", lw=3)
            ax.set_xlim((x0, x1))

        if re.search("rain|runoff", vartxt):
            d = int(re.search("[0-9]+", vartxt).group())//24
            n = re.sub("_", " ", re.sub("_.*", "", vartxt))
            txt = f"Maximum {n} total - {d} days"
        else:
            txt = "Specific peak flow"

        title = f"({letters[iax]}) {txt}"
        ax.set(title=title)

        ax.legend(loc=4, fontsize="large", framealpha=0.8)

        if re.search("specificflow_peak", vartxt):
            ax.set_ylim((5e-3, 1e2))

        elif re.search("runoff_", vartxt):
            _, y1 = ax.get_ylim()
            ax.set_ylim((10, y1))

    fp = fimg / f"FIGB_scatterplots.{imgext}"
    fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    main()
