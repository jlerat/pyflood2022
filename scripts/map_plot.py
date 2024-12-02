#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-03-15 14:14:02.600596
## Comment : Figure showing rainfall map
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path
from collections import OrderedDict
from string import ascii_letters as letters

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import shapefile

from netCDF4 import Dataset, num2date

from scipy.stats import percentileofscore
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter, \
                            maximum_filter

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import ConnectionPatch
import matplotlib.patheffects as patheff
from matplotlib import ticker

def get_shortname(name):
    """ Get short station name """
    s = re.sub(" \(.*", "", name)
    if re.search(" (a|A)t ", name):
        s = re.sub(".* (a|A)t ", "", s)
    else:
        s = re.sub(" .*", "", s)

    return s.title()


def plot_rivers(ax, fshp_rivers, *args, **kwargs):
    lines = {}
    kwargs["lw"] = kwargs.get("lw", 0.4)
    kwargs["color"] = kwargs.get("color", "#00a9ce")
    name_filter = kwargs.get("name_filter", ".*")
    if "name_filter" in kwargs:
        kwargs.pop("name_filter")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    with shapefile.Reader(str(fshp_rivers), "r") as shp:
        nsh = len(shp.shapes())
        for ish, shrec in enumerate(shp.shapeRecords()):
            dd = shrec.record.as_dict()
            name = dd["NAME"]
            if not re.search(name_filter, name):
                continue

            line = np.ascontiguousarray(np.array(shrec.shape.points))
            ax.plot(line[:, 0], line[:, 1], *args, **kwargs)

            lines[name] = ax.get_lines()[-1]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return lines


def plot_coast(ax, fshp_coast, *args, **kwargs):
    kwargs["lw"] = kwargs.get("lw", 1.)
    kwargs["color"] = kwargs.get("color", "k")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    with shapefile.Reader(str(fshp_coast), "r") as shp:
        nsh = len(shp.shapes())
        for ish, shrec in enumerate(shp.shapeRecords()):
            line = np.ascontiguousarray(np.array(shrec.shape.points))
            ax.plot(line[:, 0], line[:, 1], *args, **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_cities(ax, cities, text_kwargs={}):
    # Plot options
    plot_kwargs = {
        "marker": "s", \
        "mfc": "tab:orange", \
        "mec" : "k", \
        "ms": 7, \
        "color": "none"
    }
    text_kwargs["color"] = text_kwargs.get("color", "black")
    pe = [patheff.withStroke(linewidth=2, foreground="w")]
    text_kwargs["path_effects"] = text_kwargs.get("path_effects", pe)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for icity, (city, xy) in enumerate(cities.items()):
        lab = "Capital city" if icity == 0 else ""
        lines = ax.plot(*xy, label=lab, **plot_kwargs)
        txt = ax.annotate(city, xy, **text_kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def main():
    #----------------------------------------------------------------------
    # @Configuration
    #----------------------------------------------------------------------
    # Image extension
    imgext = "png"

    # plot mosaic
    sids = ["138110", "202001", "203014", "203010"]
    mosaic = [["grid_sm", "grid_rain", f"tsq_{sid}"] for sid in sids]

    # Rainfall aggregation
    durations = [120] #[24, 72, 120]

    # time slice
    start = pd.to_datetime("2022-02-20 07:00")
    end = pd.to_datetime("2022-03-10 08:00")

    time_awra = pd.to_datetime("2022-02-23 09:00")

    # time slice to plot
    start_ts_plot = "2022-02-23"
    end_ts_plot = "2022-03-02"

    # spatial slice
    x0, x1 = 152.2, 153.67
    y0, y1 = -29.7, -26.

    # plot config
    cmap_rain = "inferno_r"
    cmap_sm = "cividis_r"

    # Plotting config
    fdpi = 300
    awidth = 6
    aheight = awidth*0.9*(y1-y0)/(x1-x0)/len(mosaic) # Equal scale

    cities_below_kwargs = dict(
        path_effects=[patheff.withStroke(linewidth=4, foreground="w")], \
        textcoords="offset pixels", \
        ha="center", \
        fontsize=15, \
        xytext=(0, -80)
    )

    cities_top_kwargs = cities_below_kwargs.copy()
    cities_top_kwargs["xytext"] = (-5, 40)

    #----------------------------------------------------------------------
    # @Folders
    #----------------------------------------------------------------------
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent

    fsrc = source_file.parent.parent / "data"

    fimg = froot / "images" / "map"
    fimg.mkdir(exist_ok=True, parents=True)
    for f in fimg.glob("*.png"):
        f.unlink()

    fshp_rivers = fsrc / "main_rivers_NSW+QLD_simplified4.shp"

    fshp_coast = fsrc / "ne_10m_admin_0_countries_australia.shp"

    #----------------------------------------------------------------------
    # @Get data
    #----------------------------------------------------------------------
    # Towns
    towns = pd.read_csv(fsrc / "main_towns.csv", skiprows=8)
    idx = (towns.xcoord>=x0)&(towns.xcoord<=x1)\
            & (towns.ycoord>=y0)&(towns.ycoord<=y1)\
            & (towns.POPULATION_MIN>=10000)\
            & (towns.NAME!="Nambour")
    towns = towns.loc[idx]
    towns = {re.sub(" (\(|-).*", "", t.NAME): \
                        (t.xcoord, t.ycoord) for _, t in towns.iterrows()}

    towns_top = {tn:to for tn, to in towns.items() \
                    if re.search("Ball", tn)}
    towns_below = {tn:to for tn, to in towns.items() \
                    if not re.search("Bong|Yarr|Byr|Ball|Gold|Sun", tn)}

    # Streamflow
    fs = fsrc / "streamflow_data_sites_info.csv"
    sites = pd.read_csv(fs, index_col="STATIONID", \
                                dtype={"STATIONID":str},
                                skiprows=8)
    flows = {}
    for aname in [n for l in mosaic for n in l]:
        if aname.startswith("grid"):
            continue
        siteid = re.sub(".*_", "", aname)
        f = fsrc / f"streamflow_data_{siteid}.csv"
        df = pd.read_csv(f, index_col=0, parse_dates=True, \
                            skiprows=8)
        se = df.loc[:, "STREAMFLOW[m3/sec]"]
        flows[siteid] = se

    # -- AWRA --
    fnc = fsrc / "awral_data.nc"
    with Dataset(fnc, "r") as nc:
        argrid = nc["awral_data"][:].filled()
        arvarnames = nc["variable"][:].tolist()
        artimes = nc["time"]
        artimes = pd.DatetimeIndex(num2date(artimes[:], artimes.units, \
                            only_use_cftime_datetimes=False))
        arllons = nc["longitude"][:].filled()
        arllats = nc["latitude"][:].filled()

    # -- AWAP --
    fnc = fsrc / "awap_data.nc"
    with Dataset(fnc, "r") as nc:
        awgrid = nc["awap_daily_rainfall"][:].filled()
        awllons = nc["longitude"][:].filled()
        awllats = nc["latitude"][:].filled()

        aggmax = {}
        for dur in durations:
            vname = f"awap_rainfall_total_{dur//24}day_maximum"
            aggmax[(dur, "awap")] = (nc[vname][:].filled(), awllons, awllats)

    #----------------------------------------------------------------------
    # @Process
    #----------------------------------------------------------------------
    mosaic_array = np.array(mosaic)

    for cfg, maxdata in aggmax.items():
        dur, source = cfg

        print(f"Plotting dur={dur} - source={source}")

        plt.close("all")
        ncols, nrows = len(mosaic[0]), len(mosaic)
        fig = plt.figure(figsize=(ncols*awidth, nrows*aheight),\
                            layout="constrained")
        kw = None #dict(height_ratios=[1, 3, 1, 3])
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
        anames = ["grid_sm", "grid_rain"]
        anames += [n for n in list(axs.keys()) if not n.startswith("grid")]

        for iax, aname in enumerate(anames):
            ax = axs[aname]
            if aname.startswith("tsq"):
                siteid = re.sub("ts._", "", aname)
                if not siteid in flows:
                    continue

                # Flow
                se = flows[siteid]
                se.plot(ax=ax, lw=2)

                sinfo = sites.loc[siteid]
                qmax = se.max()
                area = sinfo["CATCHMENTAREA[km2]"]
                qsmax = qmax/area

                sname = get_shortname(sinfo.NAME)
                title = f"({letters[iax]}) {sname} ({area:0.0f} km$^2$)"
                ax.set(ylabel=r"streamflow [m$^3$ s$^{-1}$]", title=title)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

                axmap = axs["grid_rain"]
                x = sinfo["LONGITUDE[deg]"]
                y = sinfo["LATITUDE[deg]"]
                axmap.plot(x, y, "o", ms=12, \
                        markeredgecolor="w", \
                        markerfacecolor="tab:red", \
                        markeredgewidth=2, \
                        zorder=1000)

                lab = "\n".join(re.split(" ", sname))
                axmap.annotate(lab, (x, y), \
                    va="bottom", ha="center", \
                    textcoords="offset pixels", \
                    xytext=(0, 30), \
                    fontsize=15, \
                    path_effects=[patheff.withStroke(linewidth=4, foreground="w")])

                continue

            # Plot config
            if aname=="grid_rain":
                maxval, llons, llats = maxdata
                toplot = maxval

                vmin = 100*(int(np.nanmin(toplot)/100))
                vmax = 100*(int(np.nanmax(toplot)/100)+1)

                lam = 0.5
                fw, inv = lambda x: np.power(1+x, lam), lambda y: np.power(y, 1./lam)-1
                bounds = inv(np.linspace(fw(vmin), fw(vmax), 10))
                base = 100
                bounds = np.unique(np.round(bounds/base)*base)
                cmap = cmap_rain
            else:
                ii = arvarnames.index("rzsm_pfull")
                toplot = argrid[ii, artimes==time_awra, :, :].squeeze()

                # .. smooth taking into account coast
                isnan = np.isnan(toplot)
                toplot[isnan] = -999
                m = maximum_filter(toplot, 10)
                toplot[isnan] = m[isnan]
                toplot = gaussian_filter(toplot, 0.7)
                toplot[isnan] = np.nan

                vmin = 0.
                vmax = 1

                bmin = 0.5
                bounds = [0.]+[bmin+0.1*k for k in range(100) if bmin+0.1*k<=1]
                llons, llats = arllons, arllats
                cmap = cmap_sm


            norm = BoundaryNorm(bounds, 256)
            levels = np.linspace(vmin, bounds[-1], 200)

            # .. coast
            plot_coast(ax, fshp_coast)

            # .. rivers
            plot_rivers(ax, fshp_rivers, lw=1.5, color="0.1")

            if aname == "grid_sm":
                backcol = dict(Wilsons="tab:green", Richmond="tab:orange",\
                        Mary="tab:pink")
                for rn in ["Wilsons", "Richmond", "Mary"]:
                    fn = f"RIVERS|{rn}"
                    backc = backcol[rn]
                    plot_rivers(ax, fshp_rivers, name_filter=rn, lw=4, color=backc)
                    plot_rivers(ax, fshp_rivers, name_filter=rn, lw=1.5, color="k")
                    ax.plot([], [], "-", lw=4, color=backc, label=f"{rn} River")

            # .. towns
            if aname == "grid_sm":
                plot_cities(ax, cities=towns_top, text_kwargs=cities_top_kwargs)
                plot_cities(ax, cities=towns_below, text_kwargs=cities_below_kwargs)


            # .. surface data
            cnt = ax.contourf(llons, llats, toplot, \
                        cmap=cmap, norm=norm, \
                        vmin=0., vmax=vmax, \
                        levels=levels)

            for c in cnt.collections:
                c.set_edgecolor("face")

            colb = fig.colorbar(cnt, ax=ax, ticks=bounds, \
                        shrink=0.5, aspect=30, anchor=(0., 0.5))
            colb.ax.set_ylim([bounds[0], bounds[-1]])

            title = f"Rainfall\n[mm]\n" if aname=="grid_rain"\
                        else "Root\nZone\nSoil\nMoist.\n[%sat]\n"
            colb.ax.set_title(title, fontsize=12)

            ax.set_xlim((x0, x1))
            ax.set_ylim((y0, y1))

            # Decorate
            def get_ticks(a0, a1, delta=0.5, eps=1e-2):
                b0 = int(a0/delta)*delta
                b1 = (int(a1/delta)+1)*delta
                k = (b1-b0)/delta
                tk = b0+delta*np.arange(k)
                return tk[(tk>a0+eps)&(tk<b1-eps)]

            ax.set_xticks(get_ticks(x0, x1))
            ax.set_yticks(get_ticks(y0, y1-1e-3))

            if aname == "grid_rain":
                title = f"({letters[iax]}) Maximum {dur}h total rainfall\n"
            else:
                txt = time_awra.strftime("%d %b")
                title = f"({letters[iax]}) Saturation of soil column\n"\
                            +f"up to 1m depth\non {txt}"

            ax.set_title(title, fontsize=15)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

            if aname == "grid_sm":
                for art in ax.lines:
                    if re.search("Capital", art.get_label()):
                        art.set_label(None)

                ax.legend(loc=3, framealpha=1., fontsize="large")

                # Map of Australia
                axi = ax.inset_axes([0.55, 0, 0.45, 0.12])
                axi.plot([x0, x1, x1, x0, x0], \
                            [y0, y0, y1, y1, y0], "-", lw=6, color="tab:red")
                plot_coast(axi, fshp_coast, color="k", lw=1)
                axi.set(xticks=[], yticks=[])
                axi.axis("equal")
            else:
                ax.set_yticks([])

        fp = fimg / f"FIGA_{source}_rainfall_{dur}h.{imgext}"
        fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    main()
