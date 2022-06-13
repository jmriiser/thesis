import pandas as pd
import numpy as np
from src.utils import drop_missing_cols

RED_FLAGS = [
    'SubjectId',
    'DMAGE', 'DMSEX',
    'PHDIAG',
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN', 'PHPERFYN',
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'ECHLVIDD', 'ECHLVIDS', 'ECHIVSD', 'ECHLVPW',
    'ECHLADIA', 'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE',
    'DUOTHBIYN', 'DUSCYN', 'DUSC', 'DUSCGR'
]


MH = ['PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10']


CONT_VAR = [
    'DMAGE',
    'ECHLVIDD', 'ECHLVIDS',
    'ECHIVSD', 'ECHLVPW',
    'ECHLADIA',
    'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE',
    'K/L Ratio'
]

R_CONT_VAR = [
    'DMAGE',
    'ECHLVIDD', 'ECHLVIDS',
    'ECHIVSD', 'ECHLVPW',
    'ECHLADIA',
    'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',  'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE',
    #'K/L Ratio'
]


CAT_VAR = [
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN',
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'DUOTHBIYN', 'DPD'
]


R_CAT_VAR = [
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN',
    'DMSEX', 'PHDIAG',
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'DUOTHBIYN', 'DUSCYN', 'DUSC', 'DUSCGR'
]


def dpd_transform(dataset):
    for i, row in dataset.iterrows():
        if row['DUSCYN'] == 'Nei':
            dataset.at[i, ['DPD']] = 'Ikke testet'
        elif row['DUSCYN'] == 'Ukjent':
            dataset.at[i, ['DPD']] = 'Ukjent'
        else:
            if row['DUSC'] == 'Normal':
                dataset.at[i, ['DPD']] = 'Normal'
            else:
                # Perugini grade
                dataset.at[i, ['DPD']] = str(row['DUSCGR']) if str(
                    row['DUSCGR']) != 'nan' else 'Patologisk, uten grad'

    dataset = dataset.drop(columns=['DUSC', 'DUSCYN', 'DUSCGR'])
    return dataset


def get_data_understanding(dataset, sheet_name):
    import re
    raw = pd.read_excel(dataset, sheet_name=sheet_name, skiprows=1)
    raw = raw.loc[:, ~raw.columns.str.startswith('OPP')]

    # Initial Feature Selection
    included_cols = [col for col in raw.columns if any(
        feature in col for feature in RED_FLAGS)]
    rf_data = raw.drop(
        columns=[col for col in raw if col not in included_cols])
    rf_data = rf_data.loc[:, ~rf_data.columns.str.endswith('CD')]

    # Replace missing values for categorical features with nunique == 1
    for col in rf_data.columns:
        if rf_data[col].nunique() == 1 and rf_data[col].dtype != np.float64:
            rf_data[[col]] = rf_data[[col]].fillna('Ukjent')

    # Drop features with >=90 missing values -> Rename columns
    data = drop_missing_cols(rf_data, 90)
    #data = rf_data
    data.columns = [re.sub('.*\.', '', col) for col in data.columns]
    data = data.drop(
        columns=[col for col in data.columns if col not in RED_FLAGS])

    data = data[data['PHDIAG'].notna()]
    data = data[data.DMAGE > 0]
    #data = dpd_transform(data)

    return data


def get_raw_data(dataset, sheet_name):
    import re
    raw = pd.read_excel(dataset, sheet_name=sheet_name, skiprows=1)
    raw = raw.loc[:, ~raw.columns.str.startswith('OPP')]

    # Initial Feature Selection
    included_cols = [col for col in raw.columns if any(
        feature in col for feature in RED_FLAGS)]
    rf_data = raw.drop(
        columns=[col for col in raw if col not in included_cols])
    rf_data = rf_data.loc[:, ~rf_data.columns.str.endswith('CD')]

    # Replace missing values for categorical features with nunique == 1
    for col in rf_data.columns:
        if rf_data[col].nunique() == 1 and rf_data[col].dtype != np.float64:
            rf_data[[col]] = rf_data[[col]].fillna('Ukjent')

    # Drop features with >=90 missing values -> Rename columns
    data = drop_missing_cols(rf_data, 90)
    data.columns = [re.sub('.*\.', '', col) for col in data.columns]
    data = data.drop(
        columns=[col for col in data.columns if col not in RED_FLAGS])

    data = data[data['PHDIAG'].notna()]
    data = data[data.DMAGE > 0]
    data = dpd_transform(data)

    data['K/L Ratio'] = data['HEKAPRE'] / data['HELAMRE']
    data = data.drop(columns=['HEKAPRE', 'HELAMRE'])

    return data


def get_data(dataset, sheet_name):
    raw = pd.read_excel(dataset, sheet_name=sheet_name,
                        skiprows=1)  # Skip first row
    raw.columns = [col.replace('1.', '')
                   for col in raw.columns]  # Remove 1. from col names
    raw[MH] = (raw[MH].notnull()).astype('int')

    # Feature Selection
    rf_data = raw.drop(columns=[col for col in raw if col not in RED_FLAGS])

    # Group data by patient and select last available value for columns
    df = rf_data.groupby('SubjectId').last()
    # Missing data - Drop all rows with no diagnosis
    df = df[df['PHDIAG'].notna()]

    # Missing data - Replace missing with "Ukjent"
    df[['EKGVOLT']] = df[['EKGVOLT']].fillna('Ukjent')
    df[['DUSCYN']] = df[['DUSCYN']].fillna('Ukjent')
    df[['DUOTHBIYN']] = df[['DUOTHBIYN']].fillna('Ukjent')
    df[['PHSPIYN']] = df[['PHSPIYN']].fillna('Ukjent')
    df[['PHCARPYN']] = df[['PHCARPYN']].fillna('Ukjent')
    df[['PHSYNA']] = df[['PHSYNA']].fillna('Ukjent')

    # Feature Engineering
    df = dpd_transform(df)

    df['K/L Ratio'] = df['HEKAPRE'] / df['HELAMRE']
    df = df.drop(columns=['HEKAPRE', 'HELAMRE'])

    return df


if __name__ == '__main__':
    pass
