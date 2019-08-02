import pandas as pd
import numpy as np

DATA_PATH = '../data/'
FILENAME_TRIP = 'trip.csv.zip'
FILENAME_STATION = 'station.csv'


def distance(lat_x, long_x, lat_y, long_y):
    lat_x = np.deg2rad(lat_x)
    lat_y = np.deg2rad(lat_y)
    long_x = np.deg2rad(long_x)
    long_y = np.deg2rad(long_y)
    # great-circle distance using haversine formula and WGS-84
    dis = 2 * np.arcsin(np.sqrt(np.square(np.sin(lat_y - lat_x)) / 2) +
                        np.cos(lat_x) * np.cos(lat_y) * np.sqrt(np.square(np.sin(long_y - long_x)) / 2)) * 6371.009
    return dis


def cal_trip_distance(file_trip=DATA_PATH + FILENAME_TRIP, file_station=DATA_PATH + FILENAME_STATION, nrows=None):
    df_trip = pd.read_csv(file_trip, usecols=['id', 'start_station_id', 'end_station_id'], nrows=nrows)
    df_station = pd.read_csv(file_station, usecols=['id', 'lat', 'long'])
    df_station.rename(columns={'id': 'station_id'}, inplace=True)

    df_trip = df_trip.merge(df_station, how='left', left_on='start_station_id', right_on='station_id')
    df_trip = df_trip.merge(df_station, how='left', left_on='end_station_id', right_on='station_id')
    df_trip.drop(['start_station_id', 'end_station_id', 'station_id_x', 'station_id_y'], axis=1, inplace=True)

    df_trip['distance'] = distance(df_trip.lat_x, df_trip.long_x, df_trip.lat_y, df_trip.long_y)
    df_trip.drop(['lat_x', 'long_x', 'lat_y', 'long_y'], axis=1, inplace=True)
    return df_trip


if __name__ == '__main__':
    df = cal_trip_distance()
    print(df.head())
