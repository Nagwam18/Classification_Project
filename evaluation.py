from sklearn.metrics import silhouette_score
import numpy as np

def compute_silhouette_score(data, labels):
    return silhouette_score(data, labels)

def compute_purity(true_labels, predicted_labels):
    contingency_matrix = np.zeros((max(true_labels)+1, max(predicted_labels)+1))
    for true, pred in zip(true_labels, predicted_labels):
        contingency_matrix[true, pred] += 1
    return np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)
