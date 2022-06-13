from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


red_flags = [
    'SubjectSeq',
    'DMAGE', 'DMSEX', 'PHRVSPYN', 'PHDIAG', 'PHMH2CD',
    'PHMH6CD', 'PHMH7CD', 'PHMH8CD', 'PHMH9CD', 'PHMH10CD',
    'EKGVOLT', 'ECHLVIDD', 'ECHLVIDS', 'ECHIVSD', 'ECHLVPW',
    'ECHLADIA', 'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE',
    'DUOTHBIYN', 'DUSCGRCD'
]

medical_history = [f'PHMH{i}' for i in range(1, 19)]

standardized_data = [
    'DMAGE',
    'ECHLVIDD', 'ECHLVIDS',
    'ECHIVSD', 'ECHLVPW',
    'ECHLADIA',
    'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE',
    'K/L Ratio'
]

ohe_data = [
    'DMSEX', 'PHRVSPYN',
    'PHMH2CD', 'PHMH6CD', 'PHMH7CD', 'PHMH8CD', 'PHMH9CD', 'PHMH10CD',
    'EKGVOLT', 'DUOTHBIYN', 'DUSCGRCD'
]


def custom_summary(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    total_missing = df.isnull().sum()
    missing_value_df = pd.DataFrame({
        'percent_missing': percent_missing,
        'total_missing:': total_missing,
        'num_unique_vals:': df.nunique(),
        'd_type:': df.dtypes})
    return missing_value_df


def onehot(dataset, column):
    enc = OneHotEncoder(handle_unknown='ignore')
    out = dataset[column].values
    out = enc.fit_transform(out.reshape(-1, 1))

    return out.toarray()


def drop_missing_cols(df, thresh):
    columns = df.columns
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame(
        {'column_name': columns, 'percent_missing': percent_missing})

    missing_drop = list(
        missing_value_df[missing_value_df.percent_missing >= thresh].column_name)
    df = df.drop(missing_drop, axis=1)
    return df


def get_last_valid(series):
    return series.dropna().iloc[-1]


def plot_missing_percentages(dataset, thresh=0, save=False):
    import os

    plt.figure(figsize=(7, 7), dpi=160)
    plt.barh(custom_summary(dataset).sort_values(by=['percent_missing']).index.values, custom_summary(
        dataset).sort_values(by=['percent_missing']).percent_missing.values)
    if thresh != 0:
        plt.axvline(x=thresh, color='r', linestyle='-')
    # plt.xticks(rotation='vertical')
    plt.title("Features", fontsize=15)
    plt.xlabel("Percent missing")

    if bool(save):
        plt.savefig(save, transparent=False)
    plt.show()


def summarize_clusters(data):
    df = [pd.concat(
        [data.groupby(['Clusters']).get_group(i).median().T,
         data.groupby(['Clusters']).get_group(i).describe(include='object').T.freq /
         data.groupby(['Clusters']).get_group(i).describe(include='object').T['count']]
    ) for i in range(len(data.groupby(['Clusters']))) if i in data.Clusters.unique()]
    return pd.DataFrame(df).T
