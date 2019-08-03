import numpy as np
import pandas as pd


def distance(df, columns=None):
    if columns is None:
        columns = {'lat_x': 'lat_x',
                   'long_x': 'long_x',
                   'lat_y': 'lat_y',
                   'long_y': 'long_y'}

    lat_x = np.deg2rad(df[columns['lat_x']])
    lat_y = np.deg2rad(df[columns['lat_y']])
    long_x = np.deg2rad(df[columns['long_x']])
    long_y = np.deg2rad(df[columns['long_y']])

    # great-circle distance using haversine formula and WGS-84
    def haversine(x):
        hs = np.sqrt(np.square(np.sin(x)) / 2)
        return hs

    dis = 2 * np.arcsin(haversine(lat_y - lat_x) +
                        np.cos(lat_x) * np.cos(lat_y) * haversine(long_y - long_x)) * 6371.009
    return dis


def one_hot(df, column):
    df_dummies = pd.get_dummies(df[column], prefix=column)
    return df_dummies


def day_of_week(df, column):
    df_dow = df[column].dt.dayofweek
    df_dow = pd.get_dummies(df_dow, prefix=column)
    return df_dow


def peak_hour(df, column):
    df_pt = df[column].dt.hour
    df_pt = 1 * df_pt.isin([8, 9, 16, 17, 18])
    return df_pt


if __name__ == '__main__':
    df = pd.read_csv('../data/subsample.csv', parse_dates=['start_date'])
    print(df.head())
    df_onehot = one_hot(df, 'subscription_type')
    print(df_onehot.head())
    df_dow = day_of_week(df, 'start_date')
    print(df_dow.head())
    df_pt = peak_hour(df, 'start_date')
    print(df_pt.head(20))
