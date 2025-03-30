from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

def apply_kmeans(data, n_clusters=3):
    """Apply K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)
    return kmeans

def apply_hierarchical(data, n_clusters=3):
    """Apply Hierarchical clustering."""
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(data)
    linkage_matrix = linkage(data, method='ward')
    return hierarchical, linkage_matrix
