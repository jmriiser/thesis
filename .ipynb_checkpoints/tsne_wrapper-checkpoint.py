from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification

class TSNEWrapper(BaseEstimator, TransformerMixin):
    def __init__(self,n_components,random_state=None,method='exact'):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        
    def fit(self, X, y = None):
        ts = TSNE(n_components = self.n_components,
        method = self.method, init='pca', perplexity=50, n_iter=5000, n_iter_without_progress=1000, random_state = self.random_state)
        self.X_tsne = ts.fit_transform(X)
        return self

    def transform(self, X, y = None):
        return X