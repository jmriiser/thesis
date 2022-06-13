import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#from sklearn.manifold import TSNE
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import jaccard_score
from sklearn.pipeline import Pipeline
from sklearn import metrics

"""'SubjectSeq', 'DMAGE', 'DMSEX', 'PHRVSPYNCD','EKGVOLT','ECHLVIDD','ECHLVIDS','ECHIVSD','ECHLVPW',
'ECHLADIA','ECHLAAR','ECHLAVOL','ECHEJFR','ECHEA','ECHESEPT','ECHELAT','ECHEAVG',
'HEKRRE','HEGFRRE','HEBNPRE','HETNTRE','HEKAPRE','HELAMRE', 'DUOTHBIYN',
'ECHLVEDV', 'ECHLVESV', 'HEBNPUN', 'HELAMUN', 'HETNTUN', 'HEKAPUN', 'PHDIAGCD',
'PHFAMYNCD', 'PHRVSPYN'"""

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

medical_history = [f'PHMH{i}' for i in range(1,19)]

#standardized_data = ['HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE']

#'ECHLVIDS', 'ECHEJFR', 'ECHEA', 'ECHLADIA', 'ECHLAVOL'
standardized_data = [
    'DMAGE', 
    'ECHLVIDD', 'ECHLVIDS', 
    'ECHIVSD', 'ECHLVPW',
    'ECHLADIA', 
    'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    #'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', # 'HEKAPRE', 'HELAMRE',
    'K/L Ratio'
]

ohe_data = [
    #'PHDIAGCD', 
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



def apply_tsne(df, n_comp=2):
    embedded = TSNE(n_components=n_comp, learning_rate='auto', init='random').fit_transform(df)
    return embedded

def apply_pca(df, n_comp=2):
    embedded = PCA(n_components=n_comp).fit_transform(df)
    return embedded

def apply_lda(df, n_comp=2):
    embedded = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(df)
    return embedded

def onehot(dataset, column):
    enc = OneHotEncoder(handle_unknown='ignore')
    out = dataset[column].values
    out = enc.fit_transform(out.reshape(-1, 1))

    return out.toarray()

def fill_na(df, cols, val):
    df[cols] = df[cols].fillna(value=val)
    return df

def drop_missing_cols(df, thresh):
    columns = df.columns
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': columns, 'percent_missing': percent_missing})

    missing_drop = list(missing_value_df[missing_value_df.percent_missing>thresh].column_name)
    df = df.drop(missing_drop, axis=1)
    return df

def get_last_valid(series):
    return series.dropna().iloc[-1]

def get_lapp(features):
    raw_data = pd.read_excel('raw_data.xlsx', sheet_name='Data', skiprows=1) # Skip first row
    raw_data.columns = [col.replace('1.', '') for col in raw_data.columns] # Remove 1. from col names
    raw_data[medical_history] = (raw_data[medical_history].notnull()).astype('int')
    raw_data[['EKGVOLT']] = raw_data[['EKGVOLT']].fillna(value='ukjent')
    raw_data['EKGVOLT'] = onehot(raw_data, 'EKGVOLT')
    rf_data = raw_data.drop(columns=[col for col in raw_data if col not in features]) # Drop all columns not in features
    
    final_last = rf_data.groupby('SubjectSeq').last()
    
    return final_last
    
def get_male_female(dataset):
    '''DataFrameDict = {elem : pd.DataFrame for elem in dataset.DMSEX.unique()}
    for key in DataFrameDict.keys():
        DataFrameDict[key] = dataset[:][dataset.DMSEX == key]

    male = DataFrameDict['Mann']
    female = DataFrameDict['Kvinne'] '''
    male = dataset[dataset['DMSEX_Mann'] == True]
    female = dataset[dataset['DMSEX_Kvinne'] == True]
    return male, female
    
def plot_missing_percentages(dataset):
    plt.figure(figsize=(7, 7), dpi=160)
    plt.barh(custom_summary(dataset).sort_values(by=['percent_missing']).index.values, custom_summary(dataset).sort_values(by=['percent_missing']).percent_missing.values)
    #plt.xticks(rotation='vertical')
    plt.title("Features", fontsize=15)
    plt.xlabel("Percent missing")

    plt.savefig('missing.png', transparent=False)
    plt.show()

def compute_jaccard(true, pred):
    p1 = jaccard_score(true.replace({1: 1, 2: 1, 97: 0}), pred, average=None)
    p2 = jaccard_score(true.replace({1: 0, 2: 0, 97: 1}), pred,average=None)
    
    return p1, p2

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


