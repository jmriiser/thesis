import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_single(data, preds, save=False):
    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    plt.subplot(gs[0, :2])
    sc = plt.scatter(data[0][:, 0], data[0][:, 1], c=preds[0])
    plt.legend(*sc.legend_elements(), title='clusters')

    if bool(save):
        my_path = os.path.abspath(__file__)
        plt.savefig(my_path+'/'+save)
    plt.show()


def plot_b2b(data, preds, save=False):
    import matplotlib.gridspec as gridspec
    import pandas as pd

    fig = plt.figure(figsize=(24, 20))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    plt.subplot(gs[0, :2])
    sc = plt.scatter(data[0][:, 0], data[0][:, 1], c=preds[0])
    plt.title('Dimension reduction with PCA(2)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(*sc.legend_elements(), title='clusters')

    plt.subplot(gs[0, 2:])
    sc_umap = plt.scatter(data[1][:, 0], data[1][:, 1], c=preds[1])
    plt.title('Dimension reduction with PCA and t-SNE(2)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(*sc_umap.legend_elements(), title='clusters')

    plt.subplot(gs[1, 1:3])
    sc_umap = plt.scatter(data[2][:, 0], data[2][:, 1], c=preds[2])
    plt.title('Dimension reduction with PCA and UMAP(2)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(*sc_umap.legend_elements(), title='clusters')

    if bool(save):
        my_path = os.path.abspath(__file__)
        plt.savefig(my_path+'/'+save)
    plt.show()


if __name__ == '__main__':
    pass
