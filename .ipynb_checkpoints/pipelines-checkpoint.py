import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import kneed
import umap

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from sklearn.metrics import silhouette_score, jaccard_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

RED_FLAGS = [
    'SubjectSeq',
    'DMAGE', 'DMSEX', #'PHRVSPYN',
    'PHDIAG',
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN',
    'PHMH2','PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'ECHLVIDD', 'ECHLVIDS', 'ECHIVSD', 'ECHLVPW',
    'ECHLADIA', 'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA',
    'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE',
    'DUOTHBIYN', 'DUSCYN', 'DUSC', 'DUSCGR'
]


MH = ['PHMH2','PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10']#f'PHMH{i}' for i in range(1,19)]


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
    'ECHLAAR', 'ECHLAVOL', 'ECHEJFR', 'ECHEA', 'ECHESEPT', 'ECHELAT', 'ECHEAVG',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE', 'HEKAPRE', 'HELAMRE',
]

CAT_VAR = [
    #'DMSEX',
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN',
    #'PHRVSPYN', 
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'DUOTHBIYN', 'DPD'
]

R_CAT_VAR = [
    'PHNYHA', 'PHSYNA', 'PHCARPYN', 'PHSPIYN',
    #'PHRVSPYN', 
    'DMSEX', 'PHDIAG',
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'DUOTHBIYN', 'DUSCYN', 'DUSC', 'DUSCGR'
]


DIAGCOLORS = {'AL amyloidose':'red', 'TTR amyloidose':'green', 'Annet':'blue'}


def dpd_transform(dataset):
    for i, row in dataset.iterrows():
        if row['DUSCYN'] == 'Nei':
            dataset.at[i,['DPD']] = 'Ikke testet'
        elif row['DUSCYN'] == 'Ukjent':
            dataset.at[i,['DPD']] = 'Ukjent'
        else:
            if row['DUSC'] == 'Normal':
                dataset.at[i,['DPD']] = 'Normal'
            else:
                # Perugini grade
                dataset.at[i,['DPD']] = str(row['DUSCGR']) if str(row['DUSCGR']) != 'nan' else 'Patologisk, uten grad'

    dataset = dataset.drop(columns=['DUSC', 'DUSCYN', 'DUSCGR'])
    return dataset


def get_raw_data():
    raw = pd.read_excel('raw_data.xlsx', sheet_name='Data', skiprows=1) # Skip first row
    raw.columns = [col.replace('1.', '') for col in raw.columns] # Remove 1. from col names
    
    # Feature Selection
    rf_data = raw.drop(columns=[col for col in raw if col not in R_CONT_VAR+R_CAT_VAR+['SubjectSeq']]) 
    return rf_data


def get_data():
    raw = pd.read_excel('raw_data.xlsx', sheet_name='Data', skiprows=1) # Skip first row
    raw.columns = [col.replace('1.', '') for col in raw.columns] # Remove 1. from col names
    raw[MH] = (raw[MH].notnull()).astype('int')
    
    # Feature Selection
    rf_data = raw.drop(columns=[col for col in raw if col not in RED_FLAGS]) 
    # Group data by patient and select last available value for columns
    df = rf_data.groupby('SubjectSeq').last()
    
    # Missing data - Drop all rows with no diagnosis
    df = df[df['PHDIAG'].notna()] 
    # Missing data - Replace missing with "Ukjent"
    df[['EKGVOLT']] = df[['EKGVOLT']].fillna(value='Ukjent')
    df[['DUSCYN']] = df[['DUSCYN']].fillna('Ukjent')
    #df[['PHRVSPYN']] = df[['PHRVSPYN']].fillna('Ukjent')
    df[['DUOTHBIYN']] = df[['DUOTHBIYN']].fillna(value='Ukjent')
    
    df[['PHSPIYN']] = df[['PHSPIYN']].fillna(value='Ukjent')
    df[['PHCARPYN']] = df[['PHCARPYN']].fillna(value='Ukjent')
    df[['PHSYNA']] = df[['PHSYNA']].fillna(value='Ukjent')
    
    # Feature Engineering
    df = dpd_transform(df)

    df['K/L Ratio'] = df['HEKAPRE'] / df['HELAMRE']
    df = df.drop(columns=['HEKAPRE', 'HELAMRE'])
    
    return df


def get_num_of_k(dataset):
    inertia_list = {}
    for num_clusters in np.arange(1, 21):
        km = KMeans(n_clusters=num_clusters)
        km.fit(dataset)
        inertia_list[num_clusters] = km.inertia_

    kn = kneed.KneeLocator(x=list(inertia_list.keys()), 
         y=list(inertia_list.values()), 
         curve='convex', 
         direction='decreasing'
    )
    
    plt.figure(figsize=(10, 5))
    plt.plot(inertia_list.keys(), inertia_list.values())
    plt.grid(True)
    plt.xlabel('Number of K')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    plt.savefig('elbow_plot')
    
    return kn.knee
    

cat_pipe = make_pipeline(
    OneHotEncoder(handle_unknown='ignore')
)


cont_pipe = make_pipeline(
    MinMaxScaler()
)

data_cleaner = ColumnTransformer(transformers = [
    ("categorical", cat_pipe, CAT_VAR),
    ("continuous", cont_pipe, CONT_VAR)
])


impute_pipe = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0)
)


dimension_reducer = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0),
    PCA(n_components=0.9, random_state=0),
    TSNE(n_components=2, random_state=0)
)


pca_only = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0),
    PCA(n_components=2, random_state=0),
)


umap_dr = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0),
    PCA(n_components=0.9, random_state=0),
    umap.UMAP(n_neighbors=25, min_dist=0.0, n_components=2,random_state=0)
)


preprocessor = Pipeline([
    ("data_cleaner", data_cleaner),
    ("dimension_reducer", dimension_reducer)
])


umap_preprocessor = Pipeline([
    ("data_cleaner", data_cleaner),
    ("dimension_reducer", umap_dr)
])


pca_preprocessor = Pipeline([
    ("data_cleaner", data_cleaner),
    ("dimension_reducer", pca_only)
])


shap_preprocessor = Pipeline([
    ("cat", ColumnTransformer(transformers = [
        ("categorical", cat_pipe, CAT_VAR),])),
    ("impute", impute_pipe)
])


def eval_internal_metrics(X_preprocessed, pred, X_umap, pred_umap):
    # Calculate cluster validation metrics
    score_kemans_s = silhouette_score(X_preprocessed, pred, metric='euclidean')
    score_kemans_c = calinski_harabasz_score(X_preprocessed, pred)
    score_kemans_d = davies_bouldin_score(X_preprocessed, pred)

    score_kemans_u_s = silhouette_score(X_umap, pred_umap, metric='euclidean')
    score_kemans_u_c = calinski_harabasz_score(X_umap, pred_umap)
    score_kemans_u_d = davies_bouldin_score(X_umap, pred_umap)
    
    print('Silhouette Score: %.3f' % score_kemans_s)
    print('Calinski Harabasz Score: %.3f' % score_kemans_c)
    print('Davies Bouldin Score: %.3f' % score_kemans_d)

    print('Silhouette Score: %.3f' % score_kemans_u_s)
    print('Calinski Harabasz Score: %.3f' % score_kemans_u_c)
    print('Davies Bouldin Score: %.3f' % score_kemans_u_d)


def eval_all_metrics(data, preds, preds_enc, gt):
    for X, pred, pred_enc in zip(data, preds, preds_enc):
        #cont = metrics.cluster.contingency_matrix(gt, pred_enc)
        #purity = np.sum(np.amax(cont, axis=0)) / np.sum(cont)
        
        sil = silhouette_score(X, pred, metric='euclidean')
        ch = calinski_harabasz_score(X, pred)
        db = davies_bouldin_score(X, pred)
        #jacc = jaccard_score(gt, pred_enc, average='weighted')
        acc = accuracy_score(gt, pred_enc)
        rec = recall_score(gt, pred_enc, average='weighted')
        prec = precision_score(gt, pred_enc, average='weighted')
        f1 = f1_score(gt, pred_enc, average='weighted')
        
        print('Silhouette Score: %.3f' % sil)
        print('Calinski Harabasz Score: %.3f' % ch)
        print('Davies Bouldin Score: %.3f' % db)
        print('Accuracy Score: %.3f' % acc)
        print('Recall Score: %.3f' % rec)
        print('Precision Score: %.3f' % prec)
        print('F1 Score: %.3f' % f1)
        print('\n')


def plot_b2b(data, preds, save=False):
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    
    plt.subplot(gs[0, :2])
    sc = plt.scatter(data[0][:,0], data[0][:,1], c=preds[0])
    plt.title('Dimension reduction with PCA(2)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(*sc.legend_elements(), title='clusters')

    plt.subplot(gs[0, 2:])
    sc_umap = plt.scatter(data[1][:,0], data[1][:,1], c=preds[1])
    plt.title('Dimension reduction with PCA and t-SNE(2)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(*sc_umap.legend_elements(), title='clusters')

    plt.subplot(gs[1, 1:3])
    sc_umap = plt.scatter(data[2][:,0], data[2][:,1], c=preds[2])
    plt.title('Dimension reduction with PCA and UMAP(2)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(*sc_umap.legend_elements(), title='clusters')
    
    if bool(save):
        plt.savefig(save)
    plt.show()
    
    
if __name__ == '__main__':
    pass