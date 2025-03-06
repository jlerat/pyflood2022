#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-03-24 Sun 03:43 PM
## Comment : Plot proag stats
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path
from itertools import product as prod

from string import ascii_letters as letters

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheff
import matplotlib.image as mpimg
import matplotlib.ticker as ticker

from map_plot import get_shortname

def main():
    #----------------------------------------------------------------------
    # @Config
    #----------------------------------------------------------------------
    ncols = 3

    imgext = "png"
    awidth = 7
    aheight = 6
    fdpi = 300

    cx = "PEAK_TIME_DIFF"
    cy = "ATTENUATION_UP_DOWN"
    cz = "PEAK_DOWN"

    labels = {
        "ATTENUATION_UP_DOWN": "Peak attenuation\n{uname}/{dname} [m]",
        "PEAK_TIME_DIFF": "Peak time difference\n{dname}/{uname} [hr]",
        "PEAK_DOWN": "Peak water level\n{dname} [m]",
        "RISING_SPEED_DOWN": "Water level rising speed [m/hr]"
        }

    annotation_kw = {
       "path_effects": [patheff.withStroke(linewidth=1, alpha=0.8, foreground="w")],
       "textcoords": "offset pixels",
       "ha": "center",
       "va": "bottom",
       "color": "0.2",
       "xytext": (0, 10)
       }

    #----------------------------------------------------------------------
    # @Folders
    #----------------------------------------------------------------------
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent
    fdata  = froot / "data"

    fimg = froot / "images" / "propag"
    fimg.mkdir(exist_ok=True, parents=True)
    for f in fimg.glob("*.png"):
        f.unlink()

    #----------------------------------------------------------------------
    # @Get data
    #----------------------------------------------------------------------
    fs = fdata / "sites_info.csv"
    sites_info = pd.read_csv(fs, index_col="STATIONID", skiprows=9)

    fs = fdata / "propag_data_sites_info.json"
    with fs.open("r") as fo:
        sites = json.load(fo)

    # Flood propag data
    fp = fdata / "floods" / "propag_data.csv"
    cdates = ["PEAK_DOWN_TIME", "PEAK_UP_TIME"]
    propag = pd.read_csv(fp, parse_dates=cdates,
                         dtype={"UPID": str, "DOWNID": str},
                         skiprows=9)
    pairs = propag.PAIR.unique()

    #----------------------------------------------------------------------
    # @Process
    #----------------------------------------------------------------------

    for ipair, pair in enumerate(pairs):
        print(f"Plotting {pair}")
        downid, upid = pair.split("_")

        dinfo = sites_info.loc[downid]
        dname = get_shortname(dinfo.NAME)

        uinfo = sites_info.loc[upid]
        uname = get_shortname(uinfo.NAME)

        idx = (propag.PAIR==pair) & propag.IS_VALID
        pro = propag.loc[idx].set_index("PEAK_DOWN_TIME")
        pro = pro.sort_index()

        fd = fimg / f"propag_{pair}.csv"
        pro.to_csv(fd)

        x = pro.loc[:, cx]
        y = pro.loc[:, cy]
        z = pro.loc[:, cz]

        t = pro.PEAK_TIME_DIFF
        w = pro.PEAK_DOWN
        imax = w.idxmax()
        is22max = (imax>pd.to_datetime("2022-02-25"))\
                  & (imax<pd.to_datetime("2022-03-10"))

        # Minimum water level to display flood name
        cuts = sites["flood_thresholds"].get(downid)
        if cuts is None:
            wmin_label = w.quantile(0.8)
        else:
            wmin_label = cuts[1]

        # Create plot
        plt.close("all")
        fig = plt.figure(figsize=(awidth, aheight),
                         layout="tight")
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        zmin = int(z.min())-1
        for xx, yy, zz, ww, tt, nn in zip(x, y, z, w, t,
                                          pro.MAJOR_FLOOD.astype(str)):
            # Plot stem
            col = "tab:red" if nn=="Feb22" else "0.4"
            ax.plot([xx]*2, [yy]*2, [zmin, zz], "-",
                    color=col, lw=1.2)

            ax.plot([xx], [yy], [zz],
                    linestyle="none",
                    marker="o",
                    markerfacecolor="w",
                    markersize=5,
                    markeredgecolor=col)

            ax.plot([xx], [yy], [zmin],
                    linestyle="none",
                    marker="o",
                    markerfacecolor=col,
                    markersize=2,
                    markeredgecolor=col)

            # Plot flood name
            if nn=="nan":
                continue

            if ww>wmin_label:
                mth = re.sub("[0-9]+", "", nn)
                yr = re.sub(mth, "", nn, re.IGNORECASE)

                if nn=="Feb22":
                    fz = 14
                    coltxt = "tab:red"
                    txt = f"{mth}\n{yr}"
                    rotation = 0
                    zztxt = zz+0.2
                else:
                    fz = 10
                    coltxt = "0.1"
                    txt = f"{mth} {yr}"
                    rotation = 90
                    zztxt = zz

                ax.text(xx, yy, zztxt, txt,
                        va="bottom",
                        ha="center",
                        color=coltxt,
                        rotation=rotation,
                        fontweight="bold",
                        fontsize=fz)

        _, z1 = ax.get_zlim()
        ax.set(zlim=(zmin, z1))

        fz = 10
        ax.set_xlabel("\n"+labels[cx].format(uname=uname, dname=dname),
                      fontsize=fz)
        ax.set_ylabel("\n"+labels[cy].format(uname=uname, dname=dname),
                      fontsize=fz)
        ax.set_zlabel(labels[cz].format(uname=uname, dname=dname),
                      fontsize=fz)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.view_init(elev=20., azim=-60, roll=0)

        if pair == "203402_203014":
            axi = ax.inset_axes((0.45, 0.62, 0.55, 0.42),
                                transform=fig.transFigure)
            fp = fdata / "Eltham2Woodlawn_map_v2.png"
            img = mpimg.imread(fp)
            aratio = img.shape[1] / img.shape[0]

            axi.imshow(img,
                       interpolation="gaussian",
                       extent=[0, 1, 0, 1./aratio])
            axi.axis("off")

        fp = fimg / f"propag_{pair}.{imgext}"
        fig.savefig(fp, dpi=fdpi)

if __name__ == "__main__":
    main()
