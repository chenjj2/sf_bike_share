import pandas as pd

PATH = '../data/'

FILES = {'trip': 'trip.csv.zip',
         'station': 'station.csv',
         'weather': 'weather.csv',
         'complete': 'complete.csv',
         }

DICT_CITY_ZIP = {'San Francisco': 94107,
                 'Redwood City': 94063,
                 'Palo Alto': 94301,
                 'Mountain View': 94041,
                 'San Jose': 95113,
                 }


def read_trip(path=None, files=None, n_row=None):
    if path is None:
        path = PATH
    if files is None:
        files = FILES
    df_trip = pd.read_csv(path + files['trip'],
                          parse_dates=['start_date'],
                          usecols=['id', 'duration', 'start_date', 'start_station_id', 'end_station_id',
                                   'subscription_type'],
                          nrows=n_row)
    df_trip['date'] = df_trip['start_date'].dt.date
    return df_trip


def read_station(path=None, files=None, n_row=None):
    if path is None:
        path = PATH
    if files is None:
        files = FILES
    df_station = pd.read_csv(path + files['station'],
                             usecols=['id', 'lat', 'long', 'city'],
                             nrows=n_row)
    df_station.rename(columns={'id': 'station_id'}, inplace=True)
    df_station['zip_code'] = df_station['city'].map(DICT_CITY_ZIP)
    df_station.drop(['city'], axis=1, inplace=True)
    return df_station


def read_weather(path=None, files=None, n_row=None):
    if path is None:
        path = PATH
    if files is None:
        files = FILES
    df_weather = pd.read_csv(path + files['weather'],
                             parse_dates=['date'],
                             usecols=['date', 'mean_temperature_f', 'mean_dew_point_f', 'mean_humidity',
                                      'mean_sea_level_pressure_inches', 'mean_visibility_miles', 'mean_wind_speed_mph',
                                      'max_gust_speed_mph', 'precipitation_inches', 'cloud_cover', 'zip_code'],
                             nrows=n_row)
    df_weather['date'] = df_weather['date'].dt.date
    df_weather['precipitation_inches'] = df_weather['precipitation_inches'].replace({'T': '0'})
    df_weather['precipitation_inches'] = df_weather['precipitation_inches'].astype(float)
    return df_weather


def get_complete_data(path=None, files=None, n_row=None, to_file=True):
    if path is None:
        path = PATH
    if files is None:
        files = FILES
    df_trip = read_trip(path, files, n_row)
    df_station = read_station(path, files, n_row)
    df_weather = read_weather(path, files, n_row)

    # remove trips with same start/end station
    df_trip = df_trip[df_trip['start_station_id'] != df_trip['end_station_id']]

    # merge station to trip
    df_trip = df_trip.merge(df_station, how='left', left_on='start_station_id', right_on='station_id')
    df_trip.drop(['station_id', 'start_station_id'], axis=1, inplace=True)
    df_trip = df_trip.merge(df_station, how='left', left_on='end_station_id', right_on='station_id')
    df_trip.drop(['station_id', 'end_station_id'], axis=1, inplace=True)

    # remove trips across cities
    df_trip = df_trip[df_trip['zip_code_x'] == df_trip['zip_code_y']]
    df_trip.drop(['zip_code_y'], axis=1, inplace=True)
    df_trip.rename(columns={'zip_code_x': 'zip_code'}, inplace=True)
    df_trip['zip_code'] = df_trip['zip_code'].astype(int)

    # merge weather to trip
    df_trip = df_trip.merge(df_weather, how='left', on=['date', 'zip_code'])
    df_trip.drop(['date'], axis=1, inplace=True)

    if to_file:
        df_trip.to_csv(path + files['complete'], index=False)

    return df_trip


if __name__ == '__main__':
    df = get_complete_data()

