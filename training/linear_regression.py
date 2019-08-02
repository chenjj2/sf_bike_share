from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

import pandas as pd
from preprocess.trip_distance import cal_trip_distance
from preprocess.trip_time import trip_id_to_encoded_time

DATA_PATH = '../data/'
FILENAME_TRIP = 'trip.csv.zip'
FILENAME_STATION = 'station.csv'
FILENAME_COMBINE = 'combine.csv'

file_trip = DATA_PATH + FILENAME_TRIP
file_station = DATA_PATH + FILENAME_STATION

NROWS = None

df = pd.read_csv(file_trip, usecols=['id', 'duration', 'start_station_id', 'end_station_id'], nrows=NROWS)
df_time = trip_id_to_encoded_time(file_trip, nrows=NROWS)
df_distance = cal_trip_distance(file_trip, file_station, NROWS)

df = df[df['start_station_id'] != df['end_station_id']]
df = df.merge(df_time, how='left', on='id')
df = df.merge(df_distance, how='left', on='id')
df = df.sample(n=10000)
df.to_csv(DATA_PATH + FILENAME_COMBINE, index=False)
del df_time, df_distance

df.drop(['id', 'start_station_id', 'end_station_id'], axis=1, inplace=True)
column_names = list(df.columns)
duration_index = column_names.index('duration')
X_names = column_names[:duration_index] + column_names[(duration_index + 1):]

X_train, X_test, y_train, y_test = train_test_split(df[X_names], df['duration'], test_size=0.1, random_state=42)

model = LassoCV()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.coef_)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
print(r2, mse, msle)