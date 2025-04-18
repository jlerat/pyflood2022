[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14271026.svg)](https://doi.org/10.5281/zenodo.14271026) [![Build pyflood2022](https://github.com/jlerat/pyflood2022/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/jlerat/pyflood2022/actions/workflows/python-package-conda.yml)

# pyflood2022

This package contains the data and code used in the paper 
by Lerat and Vaze (2025), Communications Earth & Environment, in review.

## Data
The data provided in this package is in the data/floods folder and includes
* [major_floods.csv](data/floods/major_floods.csv): List of major floods in
  Australia.  

* [sites_info.csv](data/sites_info.csv): List of gauging stations.

* [streamflow_data_sites_info.csv](data/streamflow_data_sites_info.csv):
  Additional information related to upstream/downstream flow propagation
  between sites.

* [flood_data_v5.zip](data/floods/flood_data_v5.zip): List of flood events and
  associated data including peak flow, maximum 10 days runoff and results of
  flood frequency analysis.

* [flood_data_censored_v5.zip](data/floods/flood_data_censored_v5.zip): Same data
  with flood frequency results using a censored fit where the lowest 20% of
  annual maximums are left censored.

* [propag_data.csv](data/floods/propag_data.csv): Data on flood propagation
  from site to site during the 2022 flood.

* [streamflow_data_138110.csv](data/floods/streamflow_data_138110.csv): Streamflow data during the 2022 
  flood for site 138110. Similar data is available for sites 202001, 203010, 203014.

* [awap_data.nc](data/floods/awap_data.nc): Gridded rainfall data during the
  2022 flood.

* [awra_v6_data.nc](data/floods/awra_v6_data.nc): Gridded soil moisture data during the
  2022 flood.

## Code
The code used to generate the figures of the paper is provided as separate
script for each figure:

* [map_plot.py](scripts/map_plot.py): Script to generate Figure 1 (map).
* [propag_plot.py](scripts/propag_plot.py): Script to generate Figure 2 (peak
  time informations).
* [scatter_plot.py](scripts/scatter_plot.py): Script to generate Figure 3 (scatter plots).
* [regional_floods_maps.py](scripts/regional_floods_maps.py): Script to generate Figure 4 (maps of regional flood events).
* [surprise_plot.py](scripts/surprise_plot.py): Script to generate Figure 5 (statistics of the surprise index).
* [regional_floods_stats.py](scripts/regional_floods_maps.py): Script to generate Figures from the supplementary material (statistics related to regional flood events).

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.

For proper citation of this project, please refer to the [CITATION.cff](CITATION.cff) file, which provides guidance on how to cite the software and relevant publications.
