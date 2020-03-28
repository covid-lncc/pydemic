import pytest

from pydemic.data_gather import AvailableCountryData


def test_available_countries():
    available_countries = AvailableCountryData()
    df_available_countries = available_countries.get_dataframe_for_available_countries

    assert df_available_countries.shape[0] == 249


def test_invalid_data_source_for_available_countries():
    invalid_data_source = "WHO"
    with pytest.raises(ValueError, match="Unavailable data source."):
        AvailableCountryData(data_source=invalid_data_source)
