#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- Script Meta Data --
# Author  : ler015
# Created : 2024-02-19 10:54:28.122946
# Comment : Compare list of major floods with collected flood events
#
# ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path
from string import ascii_letters as letters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

from map_plot import plot_shape

def main(version):
    #----------------------------------------------------------------------
    # @Config
    #----------------------------------------------------------------------
    selected_floods = [
        #"Feb74",
        #"Aug86",
        "QLD-Dec10",
        #"Jan11",
        #"Jul16",
        "Feb22"
    ]
    qcuts = [0, 0.5, 5, 10, 30]

    # Plotting config
    fdpi = 300
    awidth = 6
    aheight = 3.5

    #----------------------------------------------------------------------
    # @Folders
    #----------------------------------------------------------------------
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent

    fsrc  = froot / "data"

    fimg = froot / "images" / "regional_floods_maps"
    fimg.mkdir(exist_ok=True, parents=True)

    fshp_coast = fsrc / "gis" / "australia.shp"

    #----------------------------------------------------------------------
    # @Get data
    #----------------------------------------------------------------------
    fs = fsrc / "sites_info.csv"
    sites = pd.read_csv(fs, dtype={"STATIONID": str},
                        index_col="STATIONID",
                        comment="#")

    fe = fsrc / "floods" / f"flood_data_v{version}.zip"
    eventdata = pd.read_csv(fe, dtype={"SITEID": str},
                            comment="#")

    siteids = eventdata.SITEID.unique()
    coords = sites.loc[siteids, ["LONGITUDE[deg]", "LATITUDE[deg]"]]

    fm = froot / "data" / "floods" / f"major_floods.csv"
    mfloods = pd.read_csv(fm, index_col="FLOODID",
                          comment="#")

    pat = "|".join(selected_floods)
    iselected = mfloods.index.str.contains(pat)
    mfloods = mfloods.loc[iselected]

    #----------------------------------------------------------------------
    # @Process
    #----------------------------------------------------------------------
    colors = plt.get_cmap("Reds")(np.linspace(0.2, 1, len(qcuts)))

    plt.close("all")
    ncols = 2
    mosaic = np.array_split(mfloods.index.tolist(),
                            len(mfloods.index)//ncols)
    mosaic = [m.tolist() for m in mosaic]
    mosaic += [["legend", "legend"]]
    nrows = len(mosaic)

    fig = plt.figure(layout="constrained",
                     figsize=(ncols*awidth, nrows*aheight))
    kw = dict(height_ratios = [5] * (nrows - 1) + [1])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    for iax, (fid, ax) in enumerate(axs.items()):
        if fid == "legend":
            continue

        finfo = mfloods.loc[fid]
        ifloods = eventdata.MAJOR_FLOOD == fid
        if ifloods.sum() == 0:
            continue

        if ifloods.sum()>0:
            print(f"Drawing {fid}")
            x = eventdata.loc[ifloods, "LONGITUDE[deg]"]
            y = eventdata.loc[ifloods, "LATITUDE[deg]"]
            q = eventdata.loc[ifloods, "SPECIFICFLOW_PEAK[m3/sec/km2]"]

            ax.plot(coords.iloc[:, 0], coords.iloc[:, 1], ".",
                    color="0.7")

            for icut in range(len(qcuts)-1):
                q0, q1 = qcuts[icut:icut+2]
                idx = (q >= q0) & (q < q1)
                lab = f"[{q0:0.1f}, {q1:0.1f}["
                ax.plot(x[idx], y[idx], "o",
                        markeredgecolor="k",
                        markerfacecolor=colors[icut],
                        label=lab)

            plot_shape(ax, fshp_coast, color="k", lw=1)

            x0 = finfo["LONGITUDE_MIN"]
            x1 = finfo["LONGITUDE_MAX"]
            y0 = finfo["LATITUDE_MIN"]
            y1 = min(-10.5, finfo["LATITUDE_MAX"])
            ax.plot([x0, x1, x1, x0, x0], \
                        [y0, y0, y1, y1, y0], \
                        "-r", lw=2)

            name, dt = re.split("-", fid)
            dt = pd.to_datetime(dt, format="%b%y").strftime("%B %Y")
            name = re.sub("Australia", " Australia", name)
            name = re.sub("Rivers", " Rivers", name)
            name = re.sub("QLD", "Queensland", name)
            name = re.sub("VIC", "Victoria", name)
            title = f"({letters[iax]}) {name} - {dt}"
            ax.set_title(title, x=0.02, y=0.95, va="top", ha="left",
                         fontsize=15)

            ax.axis("equal")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

            if fid not in [m[0] for m in mosaic]:
                ax.set_yticklabels([])

            if fid in [mm for m in mosaic[:-2] for mm in m]:
                ax.set_xticklabels([])

            ax.grid(linestyle=":")

            txt = f"{len(x)} site events"
            ax.text(0.02, 0.02, txt, ha="left", va="bottom",
                    transform=ax.transAxes, fontweight="bold",
                    fontsize=11)

            txt = re.sub("/$", "", re.sub(" .*", "", finfo.URL))
            txt = re.split("/", txt)
            txt = "/".join(txt[:-1]) + f"\n/{txt[-1]}"
            ax.text(0.98, 0.02, txt, ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=10)

            # Draw legend
            if fid == mfloods.index[-1]:
                handles, labels = ax.get_legend_handles_labels()
                title = "Specific instantaneous peak flow "\
                        + r"[m$^3$.s$^{-1}$.km$^{-2}$]"
                axleg = axs["legend"]
                leg = axleg.legend(handles=handles,
                             labels=labels,
                             loc="center", title=title, ncol=len(qcuts),
                             framealpha=0, fontsize="large",
                             bbox_to_anchor=[0.5, 0.5])
                leg.get_title().set_fontsize(14)
                axleg.axis("off")

    fp = fimg / f"regional_floods_plot_v{version}.png"
    fig.savefig(fp, dpi=fdpi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map of recent regional events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-v", "--version", help="Version number",
                        type=str, default="png")
    args = parser.parse_args()
    version = args.version

    main(version)
