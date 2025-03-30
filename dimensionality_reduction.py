from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE

def apply_svd(data, n_components=2):
    """
    Apply Truncated SVD to reduce dimensionality.
    """
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(data)

def apply_pca(data, n_components=2):
    """
    Apply PCA to reduce dimensionality.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)

def apply_tsne(data, perplexity=30, learning_rate=200):
    """
    Apply t-SNE for visualization in 2D space.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    return tsne.fit_transform(data)
