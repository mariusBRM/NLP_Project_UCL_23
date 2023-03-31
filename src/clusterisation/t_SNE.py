import pandas as pd
import sys
sys.path.append("../models/no_fine_tuning/")
from model import Net, preprocessing
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import pickle

if __name__ == '__main__':

    #df_test = pd.read_csv("../../data/sentence_embeddings_no_fine_tuning_test.csv")
    #embeddings_test = df_test["embedding"].apply(lambda x : preprocessing(x))
    #labels_test = df_test["labels"].apply(lambda x : preprocessing(x))

    #embeddings_test = torch.stack(embeddings_test.to_list())
    #labels_test = torch.stack(labels_test.to_list())

    #indexes_labels_0 = np.where(labels_test != 0)
    #embeddings_test_without_zero = embeddings_test[indexes_labels_0]
    #labels_test_without_zero = labels_test[indexes_labels_0]

    #labels_test_without_zero -= 1


    import pickle

    # Open the pickle file in binary mode
    with open('../models/experiment_2/embeddings_FineTuning_multiclass_targets.pickle', 'rb') as f:
        # Load the object from the pickle file
        my_object = pickle.load(f)
        
    embeddings = []
    labels = []

    for (emb,lab) in my_object:
        #print(type(emb))
        #print(type(lab))
        embeddings.append(emb)

        labels.append(lab)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    

    tsne = TSNE(n_components=2, perplexity=10, learning_rate=300)

    #sample_test_embeddings = embeddings_test[:1000]

    X_tsne = tsne.fit_transform(embeddings)

    color_map = {1:'b', 2:'g', 3:'c', 4:'m', 5:'y', 6:'k',7:'r'}

    for i in range(len(X_tsne)):
        
        plt.scatter(X_tsne[i,0], X_tsne[i,1], c=color_map[int(labels[i])])
        
    
    plt.show()
    plt.savefig("t_sne.png")