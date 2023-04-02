from sklearn.metrics import silhouette_score, silhouette_samples
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("../models/no_fine_tuning/")
from model import preprocessing
import torch

def load_embeddings(path):
    # Open the pickle file in binary mode
    with open(path, 'rb') as f:
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

    return embeddings, labels


def calc_silhouette(embeddings, labels, path_to_save_metrics):

    df = pd.DataFrame(index = np.unique(labels), columns=['euclidean', 'cosine'])

    for m in ['euclidean', 'cosine']:

        # calculate score per class
        s_samples = silhouette_samples(embeddings, labels, metric=m)
        for lab in np.unique(labels):
            idx = np.squeeze(np.argwhere(labels == lab))
            df.loc[lab][m] = np.mean(s_samples[idx])
    
    df.to_csv(path_to_save_metrics)


    # total score
    s_euclid = silhouette_score(embeddings, labels, metric='euclidean')
    s_cosine = silhouette_score(embeddings, labels, metric='cosine')

    return s_euclid, s_cosine
    

def plots(embeddings_ft, labels_ft, embeddings_nft, labels_nft, type):
    ## method = ft (fine_tuning) or nft (no_fine_tuning)

    if type == "pca":
        pca = PCA(n_components=2)
        result_ft = pca.fit_transform(embeddings_ft)
        result_nft = pca.fit_transform(embeddings_nft)
    if type == "tsne":
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=300)
        result_ft = tsne.fit_transform(embeddings_ft)
        result_nft = tsne.fit_transform(embeddings_nft)

    list_targets = ['race', 'religion', 'origin', 'gender', 'sexuality', 'age', 'disability']
    list_labels_ft = list(map(lambda x: list_targets[int(x-1)], labels_ft))
    list_labels_nft = list(map(lambda x: list_targets[int(x-1)], labels_nft))

    fig, axs = plt.subplots(2, figsize=(8,16))
    sns.scatterplot(x=result_ft[:,0], y=result_ft[:,1], hue=list_labels_ft, palette = sns.color_palette(palette = 'colorblind',n_colors=7), ax=axs[0])
    axs[0].set_title('Fine-Tuned BERT')
    axs[0].legend_.remove()
    sns.scatterplot(x=result_nft[:,0], y=result_nft[:,1], hue=list_labels_nft, palette = sns.color_palette(palette = 'colorblind',n_colors=7), ax=axs[1])
    axs[1].set_title('Pre-Trained BERT')
    axs[1].legend_.remove()
    axs[0].legend(title='Hate Speech Target',bbox_to_anchor=(0.5, 1.1), loc='lower center', borderaxespad=0, ncols=4)
    fig.savefig(type+".png", bbox_inches='tight')



## fine tuned
embeddings, labels = load_embeddings('../models/experiment_2/embeddings_FineTuning_multiclass_targets.pickle')
s_euclid_ft, s_cosine_ft = calc_silhouette(embeddings, labels, 'silhouette_fine_tuning')


## not fine tuned


df_test = pd.read_csv("../../data/sentence_embeddings_no_fine_tuning_test.csv")
embeddings_nft = df_test["embedding"].apply(lambda x : preprocessing(x))
labels_nft = df_test["labels"].apply(lambda x : preprocessing(x))

embeddings_nft = torch.stack(embeddings_nft.to_list())
labels_nft = torch.stack(labels_nft.to_list())


s_euclid_nft, s_cosine_nft = calc_silhouette(embeddings_nft, labels_nft, 'silhouette_no_fine_tuning')


plots(embeddings_ft=embeddings, labels_ft=labels, embeddings_nft=embeddings_nft, labels_nft=labels_nft, type='pca')
plots(embeddings_ft=embeddings, labels_ft=labels, embeddings_nft=embeddings_nft, labels_nft=labels_nft, type='tsne')

avg_info = {'s_euclid_ft': s_euclid_ft,
             's_cosine_ft': s_cosine_ft,
             's_euclid_nft': s_euclid_nft,
             's_cosine_nft': s_cosine_nft
             }
df_avg_info = pd.DataFrame(avg_info, index=[0])
df_avg_info.to_csv('avg_silhouette')

