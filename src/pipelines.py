import numpy as np
import matplotlib.pyplot as plt
import kneed
from src.dataloader import CAT_VAR, CONT_VAR

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
# Needed because IterativeImputer is an experimental class
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from collections import defaultdict


class LEncoder:
    def __init__(self):
        self.encoder_dict = defaultdict(LabelEncoder)

    def fit_transform(self, data):
        encoded = data.apply(
            lambda x: self.encoder_dict[x.name].fit_transform(x))
        return encoded

    def inverse_transform(self, data):
        inversed = data.apply(
            lambda x: self.encoder_dict[x.name].inverse_transform(x))
        return inversed.astype(str)


def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def get_num_of_k(dataset, name=None):
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
    plt.plot(kn.knee, inertia_list[kn.knee],
             'bo', fillstyle='none', markersize=75)

    plt.grid(True)
    plt.xlabel('Number of K')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    if name:
        plt.savefig(f'{name}_elbow_plot')

    return kn.knee


cat_pipe = make_pipeline(
    OneHotEncoder(drop='first', handle_unknown='error')
)

cont_pipe = make_pipeline(
    MinMaxScaler()
)

impute_pipe = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0)
)

data_cleaner = ColumnTransformer(transformers=[
    ("impute", impute_pipe, CONT_VAR),
    #("categorical", cat_pipe, CAT_VAR),
    #("continuous", cont_pipe, CONT_VAR)
])

data_cleaner_cont = ColumnTransformer(transformers=[
    ("impute", impute_pipe, CONT_VAR),
    ("continuous", cont_pipe, CONT_VAR)
])

dimension_reducer = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0),
    PCA(n_components=0.9, random_state=0),
    #TSNE(n_components=2, random_state=0)
)

pca_only = make_pipeline(
    IterativeImputer(max_iter=50, initial_strategy='median', random_state=0),
    PCA(n_components=0.9, random_state=0),
)


preprocessor = Pipeline([
    ("data_cleaner", data_cleaner),
    #("dimension_reducer", dimension_reducer)
])


pca_preprocessor = Pipeline([
    ("data_cleaner", data_cleaner),
    ("dimension_reducer", pca_only)
])


shap_preprocessor = Pipeline([
    ("cat", ColumnTransformer(transformers=[
        ("categorical", cat_pipe, CAT_VAR),
        ("continuous", cont_pipe, CONT_VAR)])),
    ("impute", impute_pipe)
])


if __name__ == '__main__':
    pass
