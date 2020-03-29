"""
A module to get json data from COVID19py and translate them to pandas.DataFrame or csv files.
"""
import json
import os
from pathlib import Path
from typing import Union

import attr
from enum import Enum
import socket

import COVID19Py
import pandas as pd
from tqdm import tqdm


class DataSource(Enum):
    """
    Available data sources from COVID19Py:

    JHU: https://github.com/CSSEGISandData/COVID-19 - Worldwide Data repository operated by the
    Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE).

    CSBS: https://www.csbs.org/information-covid-19-coronavirus - U.S. County data that comes
    from the Conference of State Bank Supervisors.
    """

    JHU = 1
    CSBS = 2


class ExportFormat(Enum):
    """
    Supported file format extensions to export COVID19Py wrapped data. Available options:

    1. csv format;

    2. xlsx format (Excel native spreadsheet);

    3. JSON serialized data.
    """

    CSV = 1
    EXCEL = 2
    JSON = 3


@attr.s(auto_attribs=True)
class AvailableCountryData:
    """
    This class is responsible to manage information about available country names and provinces.

    Attributes:
    ------------
    * online_data_source: The online database form COVID19Py API to be used.

    * use_internet_connection: Set True if you want to download new data from COVID19Py API.
    """

    online_data_source: DataSource = DataSource.JHU
    use_internet_connection: bool = True
    _all_data: dict = None
    _source: str = None

    def __attrs_post_init__(self):
        if not _has_internet_connection() and self.use_internet_connection:  # pragma: no cover
            RuntimeError(
                "Internet connection is unavailable. However, internet connection is required."
            )
        if self.use_internet_connection:
            self._source = _get_online_resource_as_str(self.online_data_source)

            covid19 = COVID19Py.COVID19(data_source=self._source)
            self._all_data = covid19.getAll()
        else:
            filename = _get_absolute_path_relative_to_script("../data/all_data.json")
            with open(filename, "r") as fp:
                self._all_data = json.load(fp)

    @property
    def get_dataframe_for_available_countries(self) -> pd.DataFrame:
        """
        Get data that provides available country names. This way, we can check what data we can retrieve from COVID19Py.

        :return:
            A DataFrame containing all available countries and provinces.
        """
        country_names = list()
        country_codes = list()
        country_provinces = list()
        progress_bar = tqdm(self._all_data["locations"])
        for entry in progress_bar:
            progress_bar.set_description("Processing available countries DataFrame")
            country_name = entry["country"]
            country_names.append(country_name)

            country_code = entry["country_code"]
            country_codes.append(country_code)

            country_province = entry["province"]
            country_provinces.append(country_province)

        country_database_dict = {
            "name": country_names,
            "code": country_codes,
            "province": country_provinces,
        }

        df_available_countries = pd.DataFrame(country_database_dict)

        return df_available_countries

    def export_available_countries_data_frame(
        self, file_format: ExportFormat, file_name: Union[str, Path]
    ) -> None:
        """
        Export data gathered from COVID19Py related to available countries in data base.

        :param ExportFormat file_format:
            The format to export.

        :param str|Path file_name:
            The file name to write exported data.

        :return:
        """
        df_available_countries = self.get_dataframe_for_available_countries
        if file_format == ExportFormat.CSV:
            df_available_countries.to_csv(file_name)
        elif file_format == ExportFormat.EXCEL:
            df_available_countries.to_excel(file_name)
        elif file_format == ExportFormat.JSON:
            df_available_countries.to_json(file_name)
        else:
            ValueError("Unsupported file format to export data.")

        return

    def list_of_available_country_names(self, has_internet_connection: bool = True) -> list:
        """
        A list with all available country names.

        :param has_internet_connection:
            Set as True if you have internet connection. Otherwise set it as False.

        :return:
            The list with all available country names from database.
        """
        if has_internet_connection:
            df_country_data = self.get_dataframe_for_available_countries.copy()
            df_country_data = df_country_data["name"].drop_duplicates()
            country_names_list = list(df_country_data.values)
        else:
            filename = _get_absolute_path_relative_to_script("../data/available_countries.csv")
            df_country_data = pd.read_csv(filename)
            df_country_data = df_country_data["name"].drop_duplicates()
            country_names_list = list(df_country_data.values)

        return country_names_list

    def list_of_available_country_codes(self, has_internet_connection: bool = True) -> list:
        """
        A list with all country codes.

        :param has_internet_connection:
            Set as True if you have internet connection. Otherwise set it as False.

        :return:
            The list with all available country codes from database.
        """
        if has_internet_connection:
            df_country_data = self.get_dataframe_for_available_countries.copy()
            df_country_data = df_country_data["code"].drop_duplicates()
            country_codes_list = list(df_country_data.values)
        else:
            filename = _get_absolute_path_relative_to_script("../data/available_countries.csv")
            df_country_data = pd.read_csv(filename)
            df_country_data = df_country_data["code"].drop_duplicates()
            country_codes_list = list(df_country_data.values)

        return country_codes_list


@attr.s(auto_attribs=True)
class CountryDataCollector:
    """
    Class that provides information about COVID-19 scenario for a given country.

    Attributes:
    ------------

    * country_name: The name of the Country.

    * use_online_resources: A bool variable to turn on the use of online resources from COVID19Py.

    * online_data_source: Set which online resources from COVID19Py to use.
    """

    country_name: str
    use_online_resources: bool = False
    online_data_source: DataSource = DataSource.JHU
    _country_code: str = None

    def __attrs_post_init__(self):
        if self.use_online_resources:
            if self.online_data_source is None:
                raise ValueError(
                    "The online data source must be specified in order to use online resources."
                )

        if not type(self.country_name) is str:
            raise ValueError("Country name must be specified as a string.")

        trial_country_name = self.country_name
        if self.use_online_resources:
            available_countries = AvailableCountryData()
        else:
            available_countries = AvailableCountryData(use_internet_connection=False)

        list_available_countries = available_countries.list_of_available_country_names()
        if trial_country_name not in list_available_countries:
            raise ValueError("Queried country name is not available.")

        df_available_countries = available_countries.get_dataframe_for_available_countries
        self._country_code = self._find_country_code_from_name(
            df_available_countries, trial_country_name
        )

    @staticmethod
    def _find_country_code_from_name(
        df_available_countries: pd.DataFrame, country_name: str
    ) -> str:
        """
        Convenient method to extract a country code from a country name.

        :param df_available_countries:
            A DataFrame with all available countries and provinces.

        :param country_name:
            A country name to query.

        :return:
            Country code equivalent to the provided country name.
        """
        df_selected_country = df_available_countries[df_available_countries.name == country_name]
        df_selected_country_code = df_selected_country.code
        df_selected_country_code = df_selected_country_code.drop_duplicates()
        country_code = str(df_selected_country_code.values[0])
        return country_code

    @property
    def country_code(self) -> str:
        """
        Country code for the provided country name.

        :return:
            Country code.
        """
        return self._country_code

    @property
    def get_time_series_data_frame(self) -> pd.DataFrame:
        """
        Provide a DataFrame filled with country data, including: date, confirmed cases
        and deaths.

        :return:
            DataFrame with information for queried country.
        """
        if self.use_online_resources:
            online_resource = _get_online_resource_as_str(self.online_data_source)
            covid19 = COVID19Py.COVID19(data_source=online_resource)
            code = self._country_code
            location_dict = covid19.getLocationByCountryCode(code, timelines=True)
            if len(location_dict) == 1:
                location = location_dict[0]
                amount_of_days = len(location["timelines"]["confirmed"]["timeline"])
                days_range_list = list(range(amount_of_days))
                data_confirmed_and_deaths_dict = {
                    "day": days_range_list,
                    "date": list(location["timelines"]["confirmed"]["timeline"].keys()),
                    "confirmed": list(location["timelines"]["confirmed"]["timeline"].values()),
                    "deaths": list(location["timelines"]["deaths"]["timeline"].values()),
                }
                df_country_data = pd.DataFrame(data_confirmed_and_deaths_dict)
                df_country_data.date = df_country_data.date.astype("datetime64[ns]")
            else:
                list_of_df_province_data = list()
                progress_bar = tqdm(location_dict)
                for province_data in progress_bar:
                    progress_bar.set_description(f"Processing {self.country_name} data")
                    amount_of_days = len(province_data["timelines"]["confirmed"]["timeline"])
                    days_range_list = list(range(amount_of_days))
                    data_confirmed_and_deaths_province_dict = {
                        "province": str(province_data["province"]),
                        "day": days_range_list,
                        "date": list(province_data["timelines"]["confirmed"]["timeline"].keys()),
                        "confirmed": list(
                            province_data["timelines"]["confirmed"]["timeline"].values()
                        ),
                        "deaths": list(province_data["timelines"]["deaths"]["timeline"].values()),
                    }
                    df_province_data = pd.DataFrame(data_confirmed_and_deaths_province_dict)
                    df_province_data.date = df_province_data.date.astype("datetime64[ns]")
                    list_of_df_province_data.append(df_province_data)

                df_grouped_provinces = pd.concat(list_of_df_province_data, axis=0)
                df_grouped_provinces_sorted = df_grouped_provinces.sort_values(
                    by="province"
                ).reset_index(drop=True)
                df_country_data = (
                    df_grouped_provinces_sorted.groupby("date")["date", "confirmed", "deaths"]
                    .sum()
                    .reset_index()
                )
        else:
            filename = _get_absolute_path_relative_to_script("../data/full_dataset_jhu.csv")
            df_full_dataset_jhu = pd.read_csv(filename)
            df_grouped_country = df_full_dataset_jhu[
                df_full_dataset_jhu["country"] == self.country_name
            ].reset_index()
            df_country_data = (
                df_grouped_country.groupby("date")["date", "confirmed", "deaths"]
                .sum()
                .reset_index()
            )

        return df_country_data


def _has_internet_connection() -> bool:  # pragma: no cover
    """
    Convenient method to check if internet connection is available.

    :return:
        Internet connection status. If on, it return True. Otherwise, returns False.
    """
    try:
        # connect to the host -- tells us if the host is actually
        # reachable
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        pass
    return False


def _get_online_resource_as_str(resource: DataSource) -> str:
    """
    Convenient function to translate a DataSource enum to str compatible with COVID19Py.

    :param resource:
        An online DataSource enum for COVID19Py.

    :return:
        Equivalent string for DataSource input.
    """
    if resource == DataSource.JHU:
        return "jhu"
    elif resource == DataSource.CSBS:
        return "csbs"
    else:
        raise ValueError("Unavailable data source.")


def _get_absolute_path_relative_to_script(target_relative_path: Union[str, Path]):
    """
    Convenient function to get absolute paths in relation to this caller function.
    Useful to read data files.

    :param target_relative_path:
        Relative path in relation to the present module.

    :return:
        The equivalent absolute path.
    """
    dirname = os.path.dirname(__file__)
    filename_absolute = os.path.join(dirname, target_relative_path)
    return filename_absolute
