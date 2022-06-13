from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS
import prince
from sklearn.manifold import TSNE
import gower
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
import os
from src.eval_metrics import eval_external_metrics
from src.dataloader import get_raw_data
from src.pipelines import LEncoder, variance_threshold_selector
from src.utils import plot_missing_percentages, drop_missing_cols, custom_summary
from src.pipelines import get_num_of_k
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)


script_dir = os.path.dirname('')
results_dir = os.path.join(script_dir, 'cluster_output/')
os.makedirs(results_dir, exist_ok=True)

DIAGCOLORS = {'AL amyloidose': 'red',
              'TTR amyloidose': 'green', 'Annet': 'blue'}

cats = [
    'DMSEX',
    # 'PHNYHA',
    'PHSYNA', 'PHCARPYN',
    'PHSPIYN', 'PHPERFYN',  # 'PHOTHYN',
    'PHMH2', 'PHMH6', 'PHMH7', 'PHMH8', 'PHMH9', 'PHMH10',
    'EKGVOLT', 'DUOTHBIYN',
    'DPD'
]

conts = [
    'DMAGE',
    'ECHLVIDD',  # 'ECHLVIDS',
    'ECHIVSD', 'ECHLVPW',
    # 'ECHLADIA',
    'ECHLAAR',  # 'ECHLAVOL',
    'ECHEJFR',
    # 'ECHEA',
    'HEKRRE', 'HEGFRRE', 'HEBNPRE', 'HETNTRE',
    'K/L Ratio'
]


def summarize_clusters(data):
    df = [pd.concat(
        [data.groupby(['Clusters']).get_group(i).median().T,
         data.groupby(['Clusters']).get_group(i).describe(include='object').T.freq /
         data.groupby(['Clusters']).get_group(i).describe(include='object').T['count']]
    ) for i in range(len(data.groupby(['Clusters']))) if i in data.Clusters.unique()]
    return pd.DataFrame(df).T


def result_transformer(data):
    data_dict = {}
    for i in range(num_k):
        data_dict[i] = data.copy()
        data_dict[i]['Clusters'] = data_dict[i]['Clusters'].map(
            lambda c: 1 if c == i else 0)
    return data_dict


if __name__ == '__main__':
    # Load Data
    from sklearn.model_selection import train_test_split
    data = get_raw_data(dataset='./data/data_v2.xlsx', sheet_name='Data')
    data = drop_missing_cols(data, 40)

    X = data.copy()
    X[cats] = X[cats].fillna('Ukjent')
    X[cats] = X[cats].astype(str)
    y = data.PHDIAG

    X, X_test, y, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)

    y_ca = y.replace('TTR amyloidose', 'CA')
    y_ca = y_ca.replace('AL amyloidose', 'CA')
    y_ca = y_ca.replace('Annet', 'non-CA')

    y_ca_test = y_test.replace('TTR amyloidose', 'CA')
    y_ca_test = y_test.replace('AL amyloidose', 'CA')
    y_ca_test = y_test.replace('Annet', 'non-CA')

    from sklearn.impute import IterativeImputer

    # Prepare data
    # Continuous
    imputer = IterativeImputer(BayesianRidge())
    imputer.fit(X[conts])

    impute_data = pd.DataFrame(imputer.transform(
        X[conts]), index=X.index, columns=conts)
    normalizer = MinMaxScaler()
    normalizer.fit(impute_data)
    normalized = pd.DataFrame(normalizer.transform(
        impute_data), index=X.index, columns=conts)
    conts_high_variance = variance_threshold_selector(
        normalized, .99 * (1 - .99))
    post_hv_conts = conts_high_variance.columns

    # Categorical
    le = LEncoder()
    encoded = le.fit_transform(X[cats])
    cats_high_variance = variance_threshold_selector(encoded, .75 * (1 - .75))
    inverse_cats = le.inverse_transform(cats_high_variance)
    post_hv_cats = inverse_cats.columns

    new_cats = [*inverse_cats.columns]
    new_conts = [*conts_high_variance.columns]

    updated_data = pd.concat([conts_high_variance, inverse_cats], axis=1)
    cat_cols = list(updated_data.columns.get_indexer(new_cats))

    # Test set, here we only transform
    impute_data_test = pd.DataFrame(imputer.transform(
        X_test[conts]), index=X_test.index, columns=conts)
    normalized_test = pd.DataFrame(normalizer.transform(
        impute_data_test), index=X_test.index, columns=conts)
    conts_hv_test = normalized_test[post_hv_conts]

    cat_cols_test = X_test[post_hv_cats]

    new_cats_test = [*cat_cols_test.columns]
    new_conts_test = [*conts_hv_test.columns]

    updated_data_test = pd.concat([conts_hv_test, cat_cols_test], axis=1)
    cat_cols_test = list(updated_data_test.columns.get_indexer(new_cats_test))

    # Data Preparation
    gd = gower.gower_matrix(updated_data)
    gd_test = gower.gower_matrix(updated_data_test)
    gow_tsne = TSNE(n_components=2, perplexity=30,
                    random_state=0).fit_transform(gd)
    gow_tsne_test = TSNE(n_components=2, perplexity=30,
                         random_state=0).fit_transform(gd_test)

    famd = prince.FAMD(n_components=2, n_iter=10,
                       copy=True, check_input=True,
                       engine='auto', random_state=0)

    famd = famd.fit(updated_data)
    famd_transformed = famd.row_coordinates(updated_data)
    famd_transformed_test = famd.row_coordinates(updated_data_test)

    num_k = get_num_of_k(gd, os.path.join(results_dir, 'gower'))
    #get_num_of_k(famd_transformed, os.path.join(results_dir, 'famd'))

    # Modeling

    models = [
        KPrototypes(n_clusters=num_k, max_iter=200, n_init=15, random_state=0),
        KMeans(n_clusters=num_k, init="k-means++",
               n_init=50, max_iter=500, random_state=0),
        AgglomerativeClustering(
            n_clusters=num_k, affinity='precomputed', linkage='complete'),
        DBSCAN(eps=0.8, metric='precomputed', min_samples=5),
        OPTICS(metric='precomputed', min_samples=5)
    ]

    cols_mask = new_cats+new_conts+['PHDIAG']

    # KPrototype
    kproto_data = X.copy()[cols_mask]
    kproto_test = X_test.copy()[cols_mask]
    kproto = models[0]
    kproto_preds = kproto.fit_predict(updated_data, categorical=cat_cols)
    kproto_data['Clusters'] = kproto_preds
    kproto_test['Clusters'] = kproto.predict(
        updated_data_test, categorical=cat_cols)

    fig = plt.figure(figsize=(12, 9))
    sc = plt.scatter(famd_transformed[0], famd_transformed[1], c=kproto_preds)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(*sc.legend_elements(), title='clusters')

    plt.savefig(os.path.join(results_dir, 'gower_kproto_plot'))
    plt.show()

    data_counts = pd.DataFrame(kproto_data.groupby(
        ['Clusters', 'PHDIAG']).size()).unstack(fill_value=0)
    data_counts.plot(kind='bar', figsize=(15, 8),
                     xlabel='Clusters', ylabel='Count')
    plt.legend(['AL amyloidose', 'Annet', 'TTR amyloidose'])

    plt.savefig(os.path.join(results_dir, 'kproto_clusters'))

    # KMeans
    kmeans_data = X.copy()[cols_mask]
    kmeans_test = X_test.copy()[cols_mask]
    kmeans = models[1]
    kmeans.fit(famd_transformed)
    kmeans_preds = kmeans.predict(famd_transformed)
    kmeans_data['Clusters'] = kmeans_preds
    kmeans_test['Clusters'] = kmeans.predict(famd_transformed_test)

    # Agglomerative
    agglo_data = X.copy()[cols_mask]
    agglo_test = X_test.copy()[cols_mask]
    agglo = models[2]
    agglo_preds = agglo.fit_predict(gd)
    agglo_data['Clusters'] = agglo_preds
    agglo_test['Clusters'] = agglo.fit_predict(gd_test)

    # OPTICS
    optics_data = X.copy()[cols_mask]
    opt = models[-1]
    opt_preds = opt.fit_predict(gd)
    optics_data['Clusters'] = opt_preds
