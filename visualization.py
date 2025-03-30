import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_clusters(data, labels, title, filename):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_dendrogram(linkage_matrix, title, filename):
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def elbow_method(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.savefig("elbow_plot.png")
    plt.close()
