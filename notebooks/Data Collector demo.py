# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # How to use Data Collector module
#
# In this notebook, a simple demonstration of how to use Data Collector features is provided.

# +
import COVID19Py
import pandas as pd
from pydemic.data_collector import AvailableCountryData, CountryDataCollector
from pydemic.data_collector import (
    export_updated_full_dataset_from_jhu,
    get_updated_full_dataset_from_jhu,
)

from tqdm import tqdm

# -

# You can check the list of available countries that you can get from data collector. It can be done using internet, which will provided the up-to-date data, or you can use it offline, but the that can present several days of delay.

# +
available_countries = AvailableCountryData(use_internet_connection=True)

available_countries.list_of_available_country_names()

# +
available_countries = AvailableCountryData(use_internet_connection=False)

available_countries.list_of_available_country_names()
# -

# Likewise, you can get a `pandas.DataFrame` with recorded cases for a given country:

brazil_data = CountryDataCollector("Brazil", use_online_resources=True)

brazil_data = CountryDataCollector("Brazil", use_online_resources=False)

# Then, you can simply retrieve a `pandas.DataFrame` with cases and deaths time series for that country:

brazil_data.get_time_series_data_frame

# Finally, you can get a full and up-to-date dataset time series from JHU thanks to [COVID19Py](https://github.com/Kamaropoulos/covid19py) amazing package:

df_full_data = get_updated_full_dataset_from_jhu()

df_full_data

# And it's also possible to generate a file from the same database:

export_updated_full_dataset_from_jhu()
