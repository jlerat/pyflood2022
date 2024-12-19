[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14271026.svg)](https://doi.org/10.5281/zenodo.14271026) [![Build pyflood2022](https://github.com/jlerat/pyflood2022/actions/workflows/python-package-conda.yml/badge.svg?branch=master)](https://github.com/jlerat/pyflood2022/actions/workflows/python-package-conda.yml)

# pyflood2022

This package contains the data and code used in the paper 
by Lerat and Vaze (2024), Communications Earth & Environment, in review.

## Data
The data provided in this package is in the data/floods folder and includes
* [major_floods.csv](data/floods/major_floods.csv): List of major floods in
  Australia.  

* [sites_info.csv](data/sites_info.csv): List of gauging stations.

* [streamflow_data_sites_info.csv](data/streamflow_data_sites_info.csv):
  Additional information related to upstream/downstream flow propagation
  between sites.

* [flood_data.zip](data/floods/flood_data.zip): List of flood events and
  associated data like peak flow, maximum 5 and 10 days runoff and results of
  flood frequency analysis.

* [flood_data_censored.zip](data/floods/flood_data_censored.zip): Same data
  with flood frequency results using a censored fit where the lowest 20% of
  annual maximums are left censored.

* [propag_data.csv](data/floods/propag_data.csv): Data on flood propagation
  from site to site during the 2022 flood.

* [streamflow_data_138110.csv](data/floods/streamflow_data_138110.csv): Streamflow data during the 2022 flood for site 138110. Similar data is available for sites 202001, 203010, 203014.

* [awap_data.nc](data/floods/awap_data.nc): Gridded rainfall data during the
  2022 flood.

* [awral_data.nc](data/floods/awral_data.nc): Gridded soil moisture data during the
  2022 flood.

## Code
The code used to generate the figures of the paper are

* [map_plot.py](scripts/map_plot.py): Script to generate Figure 1 (map).
* [propag_plot.py](scripts/propag_plot.py): Script to generate Figure 2 (peak
  time informations).
* [scatter_plot.py](scripts/scatter_plot.py): Script to generate Figure 3 (scatter plots).
* [surprise_plot.py](scripts/surprise_plot.py): Script to generate Figure 4 (statistics of the surprise index).

## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.

For proper citation of this project, please refer to the [CITATION.cff](CITATION.cff) file, which provides guidance on how to cite the software and relevant publications.

TEST
