import pandas as pd
import sys
sys.path.append("../models/no_fine_tuning/")
from model import Net, preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np


if __name__ == '__main__':

    df_test = pd.read_csv("../../data/sentence_embeddings_no_fine_tuning_test.csv")
    embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    embeddings_test = torch.stack(embeddings_test.to_list())
    labels_test = torch.stack(labels_test.to_list())

    indexes_labels_0 = np.where(labels_test != 0)
    embeddings_test_without_zero = embeddings_test[indexes_labels_0]
    labels_test_without_zero = labels_test[indexes_labels_0]

    labels_test_without_zero -= 1

    

    tsne = TSNE(n_components=2, perplexity=10, learning_rate=300)

    sample_test_embeddings = embeddings_test[:1000]

    X_tsne = tsne.fit_transform(sample_test_embeddings)

    color_map = {0:'b', 1:'g', 2:'c', 3:'m', 4:'y', 5:'k', 6:'w'}

    for i in range(len(X_tsne)):
        
        plt.scatter(X_tsne[i,0], X_tsne[i,1], c=color_map[int(labels_test_without_zero[i])])
        
          
    plt.show()
    plt.savefig("t_sne.png")