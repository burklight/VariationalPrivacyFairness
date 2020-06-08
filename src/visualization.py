import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors 
import os

def plot_embeddings(embedding, embedding_s, s, figsdir):

    _plot_embedding(embedding, s, os.path.join(figsdir,'normal_reduction'))
    _plot_embedding(embedding_s, s, os.path.join(figsdir,'secret_reduction'))

def _plot_embedding(embedding, s, name_fig):

    cmap_3 = colors.ListedColormap(['red','green','blue'])

    fig, ax = plt.subplots(figsize=(6,5))
    if s is -1:
        scatter = ax.scatter(embedding[:,0],embedding[:,1], s=0.6, color='black')
    else:
        scatter = ax.scatter(embedding[:,0],embedding[:,1],c=s, s=0.6, cmap=cmap_3)
        fig.colorbar(scatter, ax=ax, boundaries=np.arange(0,4)-0.5, ticks=np.arange(0,3))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(name_fig + '.eps', format='eps')
    plt.close()
