import pandas as pd

from sklearn.preprocessing import OneHotEncoder

DATA_FILE = '/Users/jingjing/Data/sf_bike_share/trip.csv.zip'


def trip_id_to_encoded_time(file=DATA_FILE, nrows=None, output_to_file=False):

    df = pd.read_csv(file, usecols=['id', 'start_date'], nrows=nrows, parse_dates=['start_date'])
    id_ = df['id'].tolist()
    start_date_list = df['start_date'].tolist()
    start_date_features = [[x.month, x.weekday(), x.hour//4]
                           for x in start_date_list]
    enc = OneHotEncoder()
    enc.fit(start_date_features)
    encoded_start = enc.transform(start_date_features).todense().tolist()
    feature_names = enc.get_feature_names().tolist()

    data = []
    for single_id, single_start in zip(id_, encoded_start):
        data.append([single_id]+single_start)

    df_output = pd.DataFrame(data, columns=['id']+feature_names)

    if output_to_file:
        with open('trip_time.csv', mode='w') as output:
            fields_names = ['id'] + feature_names.tolist()
            writer = csv.writer(output, fieldnames=fields_names)
            writer.writeheader()
            for id, start in zip(id_, encoded_start):
                writer.writerow(dict(zip(fields_names, [id]+start)))

    return df_output


if __name__ == '__main__':
    trip_id_to_encoded_time()
    