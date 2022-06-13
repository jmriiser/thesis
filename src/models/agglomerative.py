import operator
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score

import argparse, os, sys

def silhouette_score_(matrix, outprefix, linkage):
    s_dict = {}
   
    if linkage == "ward":
        affinity="euclidean"
    else:
        affinity="l1"   
    for k in range(2,21):
        model = AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage).fit(matrix)
        s_dict[k] = silhouette_score(matrix, model.labels_)
        
    optK = max(s_dict.items(), key=operator.itemgetter(1))[0]
    lists = sorted(s_dict.items())
    x, y = zip(*lists) 
    plt.title("Agglomerative model (linkage= " + linkage + ") silhouette score \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=x,y=y,s=150,edgecolor='k')
    plt.axvline(x=optK, color='k', linestyle='--', label="optK")
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Silhouette score",fontsize=15)
    plt.xticks([i for i in range(2,21)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.savefig(outprefix+"_agglomerative" + linkage +"_silhouette_score.png")
    plt.clf()
    return optK
  
def locate_elbow(matrix, outprefix, linkage):
    if linkage == "ward":
        affinity="euclidean"
    else:
        affinity="l1"
    model = AgglomerativeClustering(affinity=affinity, linkage=linkage)
    visualizer = KElbowVisualizer(model, k=(2,50), metric='silhouette')
    visualizer.fit(matrix)
    visualizer.finalize()
    plt.savefig(outprefix+"_agglomerative_" + linkage + "_elbow.png")
    plt.clf()
    return visualizer.elbow_value_
  
def cluster_and_output(k, matrix, outprefix, linkage, preproc):
  if linkage == "ward":
      affinity="euclidean"
  else:
      affinity="l1"
      
  model = AgglomerativeClustering(n_clusters=k, affinity=affinity, linkage=linkage).fit(matrix)
  
  if preproc:
      standard_embedding = umap.UMAP(random_state=42).fit_transform(matrix)
      plt.scatter(standard_embedding[:,0],standard_embedding[:,1], c=model.labels_, cmap='plasma')
  else:
      plt.scatter(matrix[:,0],matrix[:,1], c=model.labels_, cmap='plasma')
      
  plt.suptitle("Agglomerative clustering result (linkage = {})".format(linkage))
  plt.savefig(outprefix+"_agglomerative_" + linkage + "_scatter.png")
  plt.clf()