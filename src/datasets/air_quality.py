import pandas as pd

from .real_dataset import RealDataset


class AirQuality(RealDataset):
    """
    https://archive.ics.uci.edu/ml/datasets/Air+Quality

    0 Date    (DD/MM/YYYY)
    1 Time    (HH.MM.SS)
    2 True hourly averaged concentration CO in mg/m^3 (reference analyzer)
    3 PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)
    4 True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
    5 True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
    6 PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)
    7 True hourly averaged NOx concentration in ppb (reference analyzer)
    8 PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
    9 True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
    10 PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
    11 PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)
    12 Temperature in °C
    13 Relative Humidity (%)
    14 AH Absolute Humidity
    """

    def __init__(self):
        super().__init__(
            name="AirQuality", raw_path="AirQualityUCI.csv", processed_path="air_quality.npz"
        )

    def load(self):
        raw = pd.read_csv(self.raw_path, sep=';', decimal=',')
        self._data = raw.dropna(how='all')
