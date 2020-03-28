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

# # Data Gather scratch
#
# Let's use the very nice project [COVID19Py](https://github.com/Kamaropoulos/COVID19Py) package to get `json` file with time series for COVID-19 around the world.

import COVID19Py

# And the we need to instantiate the parser (for example):

covid19 = COVID19Py.COVID19()

# We have to set which database we want. In our case, we are interest in JHU database:

covid19 = COVID19Py.COVID19(data_source="jhu")

# Now, we can get time series by country:

# +
location = covid19.getLocationByCountryCode("CA", timelines=True)

location
# -

# The above getter method returns a list. Each entry has a `json` parsed as a Python `dict`. To get the country data, we run the following:

location[0]["timelines"]

# We can get the confirmed cases with:

location[0]["timelines"]["confirmed"]["timeline"]

# Now, Deaths:

location[0]["timelines"]["deaths"]["timeline"].values()

# But how to create a `pandas.DataFrame` from this parsed data? Well, let's figure it out!

import pandas as pd

# +
amount_of_days = len(location[0]["timelines"]["confirmed"]["timeline"])
days_range_list = list(range(amount_of_days))
dict_for_a_country = {
    "day": days_range_list,
    "date": list(location[0]["timelines"]["confirmed"]["timeline"].keys()),
    "confirmed": list(location[0]["timelines"]["confirmed"]["timeline"].values()),
    "deaths": list(location[0]["timelines"]["deaths"]["timeline"].values()),
}

dict_for_a_country
# -

# Now we can put everything in a DataFrame:

# +
df_country_data = pd.DataFrame(dict_for_a_country)
df_country_data.date = df_country_data.date.astype("datetime64[ns]")

df_country_data
# -

# Not so complicated! Let's design a class to handle such kind of requests. Let's get the available countries and data for each country and province.

# +
all_data = covid19.getAll()

all_data

# +
country_names = list()
country_codes = list()
country_provinces = list()
for entry in all_data["locations"]:
    country_name = entry["country"]
    country_names.append(country_name)

    country_code = entry["country_code"]
    country_codes.append(country_code)

    country_province = entry["province"]
    country_provinces.append(country_province)

    print(f"Country name: {country_name}")
    print(f"Country code: {country_code}")
    print(f"Province: {country_province}")

    if country_province:
        print(f"Full region name: {country_name} ({country_province})")
    else:
        print(f"Full region name: {country_name}")

    print("**************************************\n")

country_database_dict = {
    "name": country_names,
    "code": country_codes,
    "province": country_provinces,
}

df_available_countries = pd.DataFrame(country_database_dict)

# +
df_available_countries = df_available_countries.sort_values(by="name").reset_index(drop=True)

df_available_countries
# -

df_available_countries.to_csv("available_countries.csv", index=False)

# +
df_from_csv = pd.read_csv("../data/available_countries.csv")

df_from_csv
# -

import attr

# @attr.s(auto_attribs=True)
# class CountryData:
