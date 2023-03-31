from sklearn.metrics import silhouette_score, silhouette_samples
import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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
    

def plots(embeddings, labels, type):

    if type == "pca":
        pca = PCA(n_components=2)
        result = pca.fit_transform(embeddings)
    if type == "tsne":
        tsne = TSNE(n_components=2, perplexity=10, learning_rate=300)
        result = tsne.fit_transform(embeddings)

    list_targets = ['race', 'religion', 'origin', 'gender', 'sexuality', 'age', 'disability']
    list_labels = list(map(lambda x: list_targets[int(x-1)], labels))

    plt.figure()
    sns.scatterplot(x=result[:,0], y=result[:,1], hue=list_labels, palette = sns.color_palette())
    plt.legend(title='Hate Speech Target',bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(type+".png", bbox_inches='tight')



# fine tuned
embeddings, labels = load_embeddings('../models/experiment_2/embeddings_FineTuning_multiclass_targets.pickle')
s_euclid_ft, s_cosine_ft = calc_silhouette(embeddings, labels, 'silhouette_fine_tuning')
plots(embeddings, labels, type='tsne')
plots(embeddings, labels, type='pca')

# # not fine tuned
# embeddings, labels = load_embeddings('')
# s_euclid_nft, s_cosine_nft = calc_silhouette(embeddings, labels, '')
# plots(embeddings, labels)

# avg_info = {'s_euclid_ft': s_euclid_ft,
#             's_cosine_ft': s_cosine_ft,
#             's_euclid_nft': s_euclid_nft,
#             's_cosine_nft': s_cosine_nft
#             }
# df_avg_info = pd.DataFrame(avg_info, index=[0])
# df_avg_info.to_csv('avg_silhouette')

