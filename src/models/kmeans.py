import time
import umap
import operator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

import argparse, os, sys

def pca_pipe(dataset, n_comp):
    pca = PCA(n_components=n_comp, random_state=0)#30)
    X = dataset
    
    # Impute NaN value data
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = IterativeImputer(max_iter=50, initial_strategy='mean', random_state=0)
    #imp = KNNImputer(n_neighbors=5, weights="distance")
    imp.fit(X)
    X_new = imp.transform(X)
    
    # Project PCA
    X_proj = pca.fit_transform(X_new)
    
    print(pca.n_components_)
    
    t_sne = TSNE(n_components=2, init='pca', perplexity=50, n_iter=5000, n_iter_without_progress=1000)
    X_proj = t_sne.fit_transform(X_proj)
    return X_proj

def calinski_harabasz(matrix, outprefix):
    d = {}
    for k in range(2, 21):
        model = KMeans(n_clusters=k).fit(matrix)
        labels = model.labels_
        d[k] = metrics.calinski_harabasz_score(matrix, labels)
    k = max(d.items(), key=operator.itemgetter(1))[0]
    lists = sorted(d.items())
    x, y = zip(*lists) 
    plt.plot(x, y, 'bx-')
    plt.xlabel('k')
    plt.ylabel('calinski_harabaz_score')
    plt.axvline(x=k, color='k', linestyle='--', label="optK")
    plt.gca().grid(True)
    plt.legend(fontsize=18)
    plt.savefig(outprefix+"_kmeans_calinski_harabasz.png")
    plt.clf()
    return k

def silhouette_score_(matrix, outprefix):
    s_dict = {}
   
    for k in range(2,21):
        model = KMeans(n_clusters=k).fit(matrix)
        s_dict[k] = silhouette_score(matrix, model.labels_)
        
    optK = max(s_dict.items(), key=operator.itemgetter(1))[0]
    lists = sorted(s_dict.items())
    x, y = zip(*lists) 
    plt.title("KMeans model silhouette score \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=x,y=y,s=150,edgecolor='k')
    plt.axvline(x=optK, color='k', linestyle='--', label="optK")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Silhouette score",fontsize=15)
    plt.xticks([i for i in range(2,21)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.savefig(outprefix+"_kmeans_silhouette_score.png")
    plt.clf()
    return optK

def locate_elbow(matrix, outprefix):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,50), metric="silhouette")
    visualizer.fit(matrix)
    visualizer.finalize()
    plt.savefig(outprefix+"_kmeans_elbow.png")
    plt.clf()
    return visualizer.elbow_value_


def custom_kmeans(k, data):
    df = get_lapp(red_flags+medical_history)
    df = df[df['PHDIAG'].notna()]
    
    df['K/L Ratio'] = df['HEKAPRE'] / df['HELAMRE']
    df = df.drop(columns=['HEKAPRE', 'HELAMRE'])
    
    # One-Hot Encode
    df = pd.get_dummies(df, columns=ohe_data)
    
    # Standardize data
    df[standardized_data] = QuantileTransformer(output_distribution='uniform', random_state=0).fit_transform(df[standardized_data])

    X = df.drop(columns=['PHDIAG', 'DMSEX', 'PHMH1', 'PHMH3','PHMH4','PHMH5','PHMH10','PHMH11','PHMH12','PHMH13','PHMH14','PHMH15','PHMH16','PHMH17','PHMH18'])
    y = df['PHDIAG']
    
    X_pca = pca_pipe(X, 0.95)
    


def cluster_and_output(k, matrix, outprefix, preproc):
    model = KMeans(n_clusters=k).fit(matrix)
    
    if preproc:
        standard_embedding = umap.UMAP(random_state=42).fit_transform(matrix)
        plt.scatter(standard_embedding[:,0],standard_embedding[:,1], c=model.labels_, cmap='plasma')
    else:
        plt.scatter(matrix[:,0],matrix[:,1], c=model.labels_, cmap='plasma')
        
    plt.suptitle("K-Means clustering result")
    plt.savefig(outprefix+"_kmeans_scatter.png")
    plt.clf()