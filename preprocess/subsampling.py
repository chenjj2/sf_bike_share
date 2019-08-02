import pandas as pd


def subsample(file, n=10000):
    df = pd.read_csv(file)
    df = df.sample(n)
    return df


if __name__ == '__main__':
    df_subsample = subsample('../data/complete.csv')
    print(df_subsample.info())
    df_subsample.to_csv('../data/subsample.csv', index=False)
