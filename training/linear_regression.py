import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ARDRegression, BayesianRidge, \
    ElasticNetCV, LassoCV, LarsCV, RidgeCV, \
    SGDRegressor, HuberRegressor, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from preprocess import feature_engineering as fe

INPUT_FILE = '~/Data/sf_bike_share/subsample.csv'
INPUT_FILE = '../data/subsample.csv'
METHODS = {'LR': LinearRegression(),
           #'ARD': ARDRegression(),
           'BayesianRidge': BayesianRidge(),
           'ElasticNet': ElasticNetCV(), 'Lasso': LarsCV(), 'Lars': LarsCV(), 'Ridge': RidgeCV(),
           "SGD": SGDRegressor(), "Huber": HuberRegressor(), "Tree": DecisionTreeRegressor(),
           #"GP": GaussianProcessRegressor(),
           "GB": GradientBoostingRegressor(), "MLP": MLPRegressor()}

def get_features():
    df = pd.read_csv(INPUT_FILE, parse_dates=['start_date'])

    print(df.head())

    # encode
    df['distance'] = fe.distance(df)
    df = pd.concat([df, fe.one_hot(df, 'subscription_type')], axis=1)
    df = pd.concat([df, fe.one_hot(df, 'zip_code')], axis=1)
    df = pd.concat([df, fe.day_of_week(df, 'start_date')], axis=1)
    df['peak_hour'] = fe.peak_hour(df, 'start_date')
    df.drop(['lat_x', 'lat_y', 'long_x', 'long_y', 'subscription_type', 'zip_code', 'start_date', 'id'], axis=1, inplace=True)
    df = df.fillna(df.mean())

    # X, y
    column_names = list(df.columns)
    duration_index = column_names.index('duration')
    X_names = column_names[:duration_index] + column_names[(duration_index + 1):]

    return df[X_names], df['duration']

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    for name, model in METHODS.items():
        print(f"Model {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{model.get_params()}")
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        #msle = mean_squared_log_error(y_test, y_pred)
        print(f"r2 {r2}, mse {mse}")

if __name__ == '__main__':
    train(*get_features())