import pytest
from pytest import fixture

from pydemic.data_collector import AvailableCountryData, ExportFormat, CountryDataCollector


@fixture
def available_countries_online():
    available_countries = AvailableCountryData()
    return available_countries


@fixture
def available_countries_offline():
    available_countries = AvailableCountryData(use_internet_connection=False)
    return available_countries


@fixture
def brazil_data():
    brazil_data = CountryDataCollector("Brazil", use_online_resources=True)
    return brazil_data


@fixture
def us_data():
    us_data = CountryDataCollector("US")
    return us_data


@fixture
def china_data():
    china_data = CountryDataCollector("China", use_online_resources=True)
    return china_data


@pytest.mark.parametrize(
    "available_countries",
    [
        pytest.lazy_fixture("available_countries_online"),
        pytest.lazy_fixture("available_countries_offline"),
    ],
)
def test_available_countries(available_countries):
    df_available_countries = available_countries.get_dataframe_for_available_countries
    assert df_available_countries.shape[0] >= 249


def test_data_regression_for_countries(available_countries_offline, data_regression):
    df_available_countries = available_countries_offline.get_dataframe_for_available_countries
    dict_available_countries = df_available_countries.to_dict()
    data_regression.check(dict_available_countries)


@pytest.mark.parametrize(
    "available_countries",
    [
        pytest.lazy_fixture("available_countries_online"),
        pytest.lazy_fixture("available_countries_offline"),
    ],
)
def test_available_countries_names_to_list(available_countries):
    list_available_countries = available_countries.list_of_available_country_names()

    assert type(list_available_countries) is list


@pytest.mark.parametrize(
    "available_countries",
    [
        pytest.lazy_fixture("available_countries_online"),
        pytest.lazy_fixture("available_countries_offline"),
    ],
)
def test_available_countries_codes_to_list(available_countries):
    list_available_country_codes = available_countries.list_of_available_country_codes()

    assert type(list_available_country_codes) is list
    assert len(list_available_country_codes) > 0


def test_invalid_data_source_for_available_countries():
    invalid_data_source = "WHO"
    with pytest.raises(ValueError, match="Unavailable data source."):
        AvailableCountryData(online_data_source=invalid_data_source)


@pytest.mark.parametrize(
    "available_countries",
    [
        pytest.lazy_fixture("available_countries_online"),
        pytest.lazy_fixture("available_countries_offline"),
    ],
)
def test_available_countries_exporter(tmp_path, available_countries):
    file_name_csv = tmp_path / "export.csv"
    file_name_xlsx = tmp_path / "export.xlsx"
    file_name_json = tmp_path / "export.json"

    available_countries.export_available_countries_data_frame(
        file_format=ExportFormat.CSV, file_name=file_name_csv
    )
    available_countries.export_available_countries_data_frame(
        file_format=ExportFormat.EXCEL, file_name=file_name_xlsx
    )

    available_countries.export_available_countries_data_frame(
        file_format=ExportFormat.JSON, file_name=file_name_json
    )

    assert len(list(tmp_path.iterdir())) == 3


def test_unavailable_country_name_in_data_collector():
    with pytest.raises(ValueError, match="Queried country name is not available."):
        CountryDataCollector("Brasil")


def test_country_code_getter(brazil_data, us_data):
    assert brazil_data.country_code == "BR"
    assert us_data.country_code == "US"


def test_data_frame_for_brazil(brazil_data):
    df_brazil_data = brazil_data.get_time_series_data_frame
    assert df_brazil_data.shape[0] > 1
    assert df_brazil_data.shape[1] > 1


def test_data_frame_for_china(china_data):
    df_china_data = china_data.get_time_series_data_frame
    assert df_china_data.shape[0] > 1
    assert df_china_data.shape[1] > 1
