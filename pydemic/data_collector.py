"""
A module to get json data from COVID19py and translate them to pandas.DataFrame or csv files.
"""
from pathlib import Path
from typing import Union

import attr
from enum import Enum
import socket

import COVID19Py
import pandas as pd


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
    """

    all_data: dict = None
    data_source: DataSource = DataSource.JHU
    _source: str = None

    def __attrs_post_init__(self):
        if not _has_internet_connection():  # pragma: no cover
            RuntimeError(
                "Internet connection is unavailable. However, internet connection is required."
            )

        if self.data_source == DataSource.JHU:
            self._source = "jhu"
        elif self.data_source == DataSource.CSBS:
            self._source = "csbs"
        else:
            raise ValueError("Unavailable data source.")

        covid19 = COVID19Py.COVID19(data_source=self._source)
        self.all_data = covid19.getAll()

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
        for entry in self.all_data["locations"]:
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

    def list_of_available_country_names(self, has_internet_connect: bool = True) -> list:
        """
        A list with all country names.

        :param has_internet_connect:
            Set true if you have internet connection. Otherwise set it as False.

        :return:
            The list with all available country names from database.
        """
        if has_internet_connect:
            df_country_data = self.get_dataframe_for_available_countries.copy()
            df_country_data = df_country_data["name"].drop_duplicates()
            country_names_list = list(df_country_data.values)
        else:
            df_country_data = pd.read_csv("../data/available_countries.csv")
            df_country_data = df_country_data["name"].drop_duplicates()
            country_names_list = list(df_country_data.values)

        return country_names_list

    def list_of_available_country_codes(self, has_internet_connect: bool = True) -> list:
        """
        A list with all country codes.

        :param has_internet_connect:
            Set true if you have internet connection. Otherwise set it as False.

        :return:
            The list with all available country codes from database.
        """
        if has_internet_connect:
            df_country_data = self.get_dataframe_for_available_countries.copy()
            df_country_data = df_country_data["code"].drop_duplicates()
            country_codes_list = list(df_country_data.values)
        else:
            df_country_data = pd.read_csv("../data/available_countries.csv")
            df_country_data = df_country_data["code"].drop_duplicates()
            country_codes_list = list(df_country_data.values)

        return country_codes_list


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
