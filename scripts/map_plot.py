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
from scipy.ndimage import gaussian_filter
from scipy.ndimage import maximum_filter

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


def plot_shape(ax, fshp, *args, **kwargs):
    lines = {}
    kwargs["lw"] = kwargs.get("lw", 0.4)
    kwargs["color"] = kwargs.get("color", "#00a9ce")
    name_filter = kwargs.get("name_filter", ".*")
    max_dist = kwargs.get("max_dist", 0.05)
    if "name_filter" in kwargs:
        kwargs.pop("name_filter")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    with shapefile.Reader(str(fshp), "r") as shp:
        nsh = len(shp.shapes())
        for ish, shrec in enumerate(shp.shapeRecords()):
            dd = shrec.record.as_dict()
            if ish==0:
                try:
                    cname = next(cn for cn in dd.keys()\
                             if re.search("name", cn, re.IGNORECASE))
                except:
                    cname = "bidule"
            name = dd.get(cname, f"shape{ish}")
            if not re.search(name_filter, name):
                continue

            line = np.ascontiguousarray(np.array(shrec.shape.points))
            dist = np.sqrt(((line[1:]-line[:-1])**2).sum(axis=1))
            starts = [[0], np.where(dist>max_dist)[0]+1, [len(line)-1]]
            starts = np.concatenate(starts)
            for istart, start in enumerate(starts[:-1]):
                end = starts[istart+1]
                ax.plot(line[start:end, 0], line[start:end, 1], *args, **kwargs)
                lines[f"{name}_{istart}"] = ax.get_lines()[-1]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return lines


def plot_cities(ax, cities, text_kwargs={}):
    # Plot options
    plot_kwargs = {
        "marker": "s",
        "mfc": "tab:orange",
        "mec" : "k",
        "ms": 7,
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
    sids = ["138110", "202001", "203014", "203402"]
    mosaic = [["grid_sm", "grid_rain", f"tsq_{sid}"] for sid in sids]

    # Rainfall aggregation
    durations = [120] #[24, 72, 120]

    # time slice
    start = pd.to_datetime("2022-02-20 07:00")
    end = pd.to_datetime("2022-03-10 08:00")

    time_awra = pd.to_datetime("2022-02-23")

    # time slice to plot
    start_ts_plot = "2022-02-23"
    end_ts_plot = "2022-03-04"

    # AWRA variable to plot
    awra_varname = "rzsm_pfull_prc"

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

    col_rivers = {
        "Wilsons": "tab:green",
        "Richmond": "tab:orange",
        "Mary": "pink"
    }

    cities_below_kwargs = dict(
        path_effects=[patheff.withStroke(linewidth=4, foreground="w")],
        textcoords="offset pixels",
        ha="center",
        fontsize=18,
        xytext=(0, -100)
    )

    cities_top_kwargs = cities_below_kwargs.copy()
    cities_top_kwargs["xytext"] = (-5, 60)

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

    fshp_rivers = fsrc / "gis" / "main_rivers_NSW+QLD_simplified4.shp"

    fshp_coast = fsrc / "gis" / "australia.shp"
    fshp_coast_nr = fsrc / "gis" / "australia_northern_rivers.shp"
    fshp_border = fsrc / "gis" / "NSW_QLD_border.shp"
    fshp_catch = fsrc / "gis" / "catchment_boundaries.shp"

    #----------------------------------------------------------------------
    # @Get data
    #----------------------------------------------------------------------
    # Towns
    towns = pd.read_csv(fsrc / "gis" / "main_towns.csv", skiprows=8)
    idx = (towns.xcoord>=x0)&(towns.xcoord<=x1)\
          & (towns.ycoord>=y0)&(towns.ycoord<=y1)\
          & (towns.POPULATION_MIN>=10000)\
          & (towns.NAME!="Nambour")
    towns = towns.loc[idx]

    towns = {re.sub(" (\(|-).*", "", t.NAME): (t.xcoord, t.ycoord)
             for _, t in towns.iterrows()}

    towns_top = {tn:to for tn, to in towns.items()
                 if re.search("Ball", tn)}
    towns_below = {tn:to for tn, to in towns.items()
                   if not re.search("Bong|Yarr|Byr|Ball", tn)}

    # Streamflow
    fs = fsrc / "streamflow_data_sites_info.csv"
    sites = pd.read_csv(fs, index_col="STATIONID",
                        dtype={"STATIONID":str},
                        skiprows=9)
    flows_waterlevels = {}
    for aname in [n for l in mosaic for n in l]:
        if aname.startswith("grid"):
            continue
        siteid = re.sub(".*_", "", aname)
        prefix = "waterlevel" if siteid == "203402" else "streamflow"
        f = fsrc / "floods" / f"{prefix}_data_{siteid}.csv"
        df = pd.read_csv(f, index_col=0, parse_dates=True, \
                            skiprows=9)

        cn = "WATERLEVEL_AHD[m]" if siteid == "203402" \
            else "STREAMFLOW[m3/sec]"
        se = df.loc[:, cn]
        flows_waterlevels[siteid] = se

    # -- AWRA --
    fnc = fsrc / "floods" / "awra_v6_data.nc"
    with Dataset(fnc, "r") as nc:
        arvarnames = nc["variable"][:].tolist()

        ivar = arvarnames.index(awra_varname)
        argrid = nc["awra_v6_daily"][ivar, :, :, :].filled()

        artimes = nc["time"]
        artimes = pd.DatetimeIndex(num2date(artimes[:], artimes.units,
                                   only_use_cftime_datetimes=False))
        arllons = nc["longitude"][:].filled()
        arllats = nc["latitude"][:].filled()

    # -- AWAP --
    fnc = fsrc / "floods" / "awap_data.nc"
    with Dataset(fnc, "r") as nc:
        awgrid = nc["awap_daily_northern_rivers"][:].filled()
        awllons = nc["longitude"][:].filled()
        awllats = nc["latitude"][:].filled()

        name = "awap_daily_northern_rivers_maxagg"
        awdurations = nc["time_aggregation"][:].filled()
        aggmax = {}
        for idur, dur in enumerate(awdurations):
            if 24 * dur not in durations:
                continue
            aggmax[(24*dur, "awap")] = (nc[name][idur, :, :].filled(),
                                        awllons, awllats)

    # --- flood data ---
    fe = fsrc / "floods" / "flood_data_censored.zip"
    eventdata = pd.read_csv(fe, dtype={"siteid": str},
                            parse_dates=["FLOW_TIME_OF_PEAK"], skiprows=9)

    #----------------------------------------------------------------------
    # @Process
    #----------------------------------------------------------------------
    mosaic_array = np.array(mosaic)

    for cfg, maxdata in aggmax.items():
        dur, source = cfg

        print(f"Plotting dur={dur} - source={source}")

        plt.close("all")
        ncols, nrows = len(mosaic[0]), len(mosaic)
        fig = plt.figure(figsize=(ncols*awidth, nrows*aheight),
                         layout="constrained")
        kw = None #dict(height_ratios=[1, 3, 1, 3])
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
        anames = ["grid_sm", "grid_rain"]
        anames += [n for n in list(axs.keys()) if not n.startswith("grid")]

        for iax, aname in enumerate(anames):
            ax = axs[aname]
            if aname.startswith("tsq"):
                siteid = re.sub("ts._", "", aname)
                if not siteid in flows_waterlevels:
                    continue

                # Flow
                se = flows_waterlevels[siteid].loc[start_ts_plot:
                                                   end_ts_plot]
                se.plot(ax=ax, lw=2)

                sinfo = sites.loc[siteid]
                qmax = se.max()
                area = sinfo["CATCHMENTAREA[km2]"]
                qsmax = qmax/area

                if siteid != "203402":
                    idx = eventdata.SITEID == siteid
                    idx &= eventdata.MAJOR_FLOOD == "NorthernRivers-Feb22"
                    finfo = eventdata.loc[idx].squeeze()
                    q100 = finfo.loc["FLOW_PEAK_C02_GEV-Q100-ALL[perc]"]
                else:
                    # See Engeny (2013), Table 4.5
                    q100 = 12.93

                x0, x1 = ax.get_xlim()
                xtxt = 0.98 * x0 + 0.02 * x1
                y0, y1 = ax.get_ylim()
                if q100 > y1:
                    ytxt = 0.98 * y1 + 0.02 * y0
                    va = "top"
                else:
                    ytxt = q100 + (y1 - y0) * 1e-2
                    ax.plot([x0, (x0 + x1) / 2], [q100]*2, "k--")
                    va = "bottom"

                if siteid == "203402":
                    txt = f"1% AEP = {round(q100, 1):0.1f} m"
                else:
                    txt = f"1% AEP = {round(q100, -1):0.0f} "+ r"m$^3$.s$^{-1}$"
                ax.text(xtxt, ytxt, txt, ha="left", va=va)

                sname = get_shortname(sinfo.NAME)
                title = f"({letters[iax]}) {sname} ({area:0.0f} km$^2$)"
                ylab = r"streamflow [m$^3$ s$^{-1}$]" if se.name.startswith("STR")\
                    else "water level [m]"

                ax.set(xlabel="", ylabel=ylab, title=title)
                ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

                axmap = axs["grid_rain"]
                x = sinfo["LONGITUDE[deg]"]
                y = sinfo["LATITUDE[deg]"]
                axmap.plot(x, y, "o", ms=10,
                           markeredgecolor="w",
                           markerfacecolor="tab:red",
                           markeredgewidth=1,
                           zorder=1000)

                # Connecting line
                imo = np.where(mosaic_array == f"tsq_{siteid}")
                imo = (imo[0][0], imo[1][0])
                xp = 0.9 if imo[1]==0 else 0.1
                yp = 0.5
                con = ConnectionPatch(\
                            xyA=(xp, yp), coordsA=ax.transAxes, \
                            xyB=(x, y), coordsB=axmap.transData, \
                            color="0.4", lw=1, zorder=10)
                fig.add_artist(con)

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
                toplot = argrid[artimes==time_awra, :, :].squeeze()

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

            # Map background
            # .. coast
            plot_shape(ax, fshp_coast_nr, color="k", lw=1.2)
            # .. NSW/QLD border
            plot_shape(ax, fshp_border, color="k", linestyle="--", lw=1.2)
            ax.plot([], [], "k--", lw=1.2, label="QLD/NSW border")

            # .. rivers
            plot_shape(ax, fshp_rivers, lw=1.5, color="0.1")

            if aname == "grid_sm":
                for rn in ["Wilsons", "Richmond", "Mary"]:
                    fn = f"RIVERS|{rn}"
                    backc = col_rivers[rn]
                    plot_shape(ax, fshp_rivers, name_filter=rn, lw=6, color=backc)
                    plot_shape(ax, fshp_rivers, name_filter=rn, lw=1.5, color="k")
                    label = f"{rn} River"
                    ax.plot([], [], "-", lw=6, color=backc, label=label)

                    if rn != "Wilsons":
                        plot_shape(ax, fshp_catch, name_filter=rn.upper(),
                                   color=backc, linestyle="-", lw=2)
                        label = f"{rn} Catchment"
                        ax.plot([], [], "-", lw=2, color=backc, label=label)

            # .. towns
            if aname == "grid_sm":
                plot_cities(ax, cities=towns_top, text_kwargs=cities_top_kwargs)
                plot_cities(ax, cities=towns_below, text_kwargs=cities_below_kwargs)

            # .. surface data
            cnt = ax.contourf(llons, llats, toplot,
                              cmap=cmap, norm=norm,
                              vmin=0., vmax=vmax,
                              levels=levels)

            for c in cnt.collections:
                c.set_edgecolor("face")

            colb = fig.colorbar(cnt, ax=ax, ticks=bounds,
                                shrink=0.5, aspect=30, anchor=(0., 0.5))
            colb.ax.set_ylim([bounds[0], bounds[-1]])

            if aname == "grid_rain":
                title = f"Rainfall\n[mm]\n"
            else:
                if awra_varname == "rzsm_pfull":
                    title = "Root\nZone\nSoil\nMoist.\n[%sat]\n"
                else:
                    title = "Prctl.\n[%]\n"
            colb.ax.set_title(title, fontsize=12)

            ax.set(xlim=(x0, x1), ylim=(y0, y1))

            # Decorate
            def get_ticks(a0, a1, delta=0.5, eps=1e-2):
                b0 = int(a0 / delta) * delta
                b1 = (int(a1 / delta) + 1) * delta
                k = (b1 - b0) / delta
                tk = b0 + delta * np.arange(k)
                return tk[(tk > a0 + eps) & (tk < b1 - eps)]

            ax.set_xticks(get_ticks(x0, x1))
            ax.set_yticks(get_ticks(y0, y1 - 1e-3))

            if aname == "grid_rain":
                title = f"({letters[iax]}) Maximum {dur}h total rainfall\n"
            else:

                txt = time_awra.strftime("%d %b")
                if awra_varname == "rzsm_pfull":
                    title = f"({letters[iax]}) Saturation of root zone soil column"\
                        + f"\non {txt}"
                else:
                    title = f"({letters[iax]}) Saturation of root zone soil column"\
                        + f"\nhistorical percentile on {txt}"

            ax.set_title(title, fontsize=15)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

            if aname == "grid_sm":
                for art in ax.lines:
                    if re.search("Capital", art.get_label()):
                        art.set_label(None)

                ax.legend(loc=3, framealpha=1., fontsize="large")

                # Map of Australia
                axi = ax.inset_axes([0.65, 0, 0.35, 0.10])
                axi.plot([x0, x1, x1, x0, x0],
                         [y0, y0, y1, y1, y0], "-", lw=6, color="tab:red")
                plot_shape(axi, fshp_coast, color="k", lw=1)
                axi.set(xticks=[], yticks=[])
                axi.axis("equal")
            else:
                ax.set_yticks([])

        fp = fimg / f"FIGA_{source}_rainfall_{dur}h.{imgext}"
        fig.savefig(fp, dpi=fdpi)


if __name__ == "__main__":
    main()
