import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def mean_vectors(embeddings, labels):

    # Compute mean vector for each class
    mean_vectors = defaultdict(list)
    for i, label in enumerate(labels):
        mean_vectors[label].append(embeddings[i])

    for label in mean_vectors:
        mean_vectors[label] = np.mean(mean_vectors[label], axis=0)

    return mean_vectors

def cosine_similarities(embeddings, labels):

    # Compute cosine similarity between predicted embeddings and mean vectors for each class
    mean_vec = mean_vectors(embeddings, labels)

    # Calculate the cosine similarity between each predicted embedding and the mean vector of its true class
    similarities = []
    for i, true_label in enumerate(true_labels):
        similarity = cosine_similarity([predicted_embeddings[i]], [mean_vec[true_label]])
        similarities.append(similarity[0][0])

    # Average cosine similarity for each class in dict
    class_similarities = defaultdict(list)
    for i, true_label in enumerate(true_labels):
        class_similarities[true_label].append(similarities[i])

    # Calculate the average cosine similarity for each class
    average_similarities = {}
    for label in class_similarities:
        average_similarities[label] = np.mean(class_similarities[label])

    return class_similarities, average_similarities, similarities



if __name__ == '__main__':
    # embeddings aleatoire et label aleatoire

    # replace with actual embeddings!!!


    
    embeddings = np.random.rand(100, 768)  # 100 example embeddings
    labels = np.random.randint(0, 8, 100)  # 100 example labels corresponding to the embeddings (classes a, b, c, d, e)

    predicted_embeddings = np.random.rand(25, 768)  # 25 example predicted embeddings
    true_labels = np.random.randint(0, 8, 25)  # 25 example true labels corresponding to the predicted embeddings


    #find the cosine similarities
    class_similarities, average_similarities, similarities = cosine_similarities(embeddings, labels)


    # Print the average cosine similarity for each class
    for label, avg_similarity in average_similarities.items():
        print(f"Class {label}: {avg_similarity:.4f}")
