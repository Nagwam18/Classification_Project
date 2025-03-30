import numpy as np
import nltk
from data_preprocessing import load_and_preprocess_people_wiki, load_and_preprocess_news
from feature_extraction import extract_features_tfidf
from dimensionality_reduction import apply_pca, apply_svd, apply_tsne
from clustering import apply_kmeans, apply_hierarchical
from evaluation import compute_silhouette_score, compute_purity
from visualization import plot_clusters, plot_dendrogram, elbow_method

#Download Required NLTK Resources
# ==========================
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Load datasets
people_wiki = load_and_preprocess_people_wiki()
news_data = load_and_preprocess_news()

# Feature extraction
X_people = extract_features_tfidf(people_wiki, ngram_range=(2, 2)).toarray()
X_news = extract_features_tfidf(news_data, ngram_range=(3, 3)).toarray()

# Dimensionality reduction
pca_people = apply_pca(X_people)
pca_news = apply_pca(X_news)
tsne_people = apply_tsne(X_people, perplexity=90, learning_rate=20)
tsne_news = apply_tsne(X_news, perplexity=90, learning_rate=20)

# Clustering
kmeans_people = apply_kmeans(tsne_people, 3)
kmeans_news = apply_kmeans(tsne_news, 3)
hierarchical_people, linkage_people = apply_hierarchical(tsne_people)
hierarchical_news, linkage_news = apply_hierarchical(tsne_news)

# Evaluation
silhouette_people_kmeans = compute_silhouette_score(tsne_people, kmeans_people.labels_)
silhouette_news_kmeans= compute_silhouette_score(tsne_news, kmeans_news.labels_)
silhouette_people_hierarc = compute_silhouette_score(tsne_people, hierarchical_people.labels_)
silhouette_news_hierarc = compute_silhouette_score(tsne_news, hierarchical_news.labels_)

news_purity_kmeans = compute_purity(news_data['category'].values, hierarchical_news.labels_)
people_purity_kmeans = compute_purity(people_wiki['clean_text'].values, hierarchical_people.labels_)
news_purity_hierarc= compute_purity(news_data['category'].values, kmeans_news.labels_)
people_purity_hierarc = compute_purity(people_wiki['clean_text'].values, kmeans_people.labels_)


# Visualization
# PCA Clustering
plot_clusters(pca_people, kmeans_people.labels_, "People Wikipedia - PCA Clusters", "pca_people_clusters.png")
plot_clusters(pca_news, kmeans_news.labels_, "NewsGroups - PCA Clusters", "pca_news_clusters.png")

# t-SNE Clustering
plot_clusters(tsne_people, kmeans_people.labels_, "People Wikipedia - t-SNE Clusters", "tsne_people_clusters.png")
plot_clusters(tsne_news, kmeans_news.labels_, "NewsGroups - t-SNE Clusters", "tsne_news_clusters.png")

# Hierarchical Clustering
plot_dendrogram(linkage_people, "People Wikipedia - Hierarchical Clustering", "dendrogram_people.png")
plot_dendrogram(linkage_news, "NewsGroups - Hierarchical Clustering", "dendrogram_news.png")

# Save results
# Open a file in write mode
with open("results.txt", "w") as f:
    f.write(f"Silhouette Score for People Wikipedia (K-Means): {silhouette_people_kmeans:.3f}\n")
    f.write(f"Silhouette Score for 20 Newsgroups (K-Means): {silhouette_news_kmeans:.3f}\n")
    f.write(f"Purity Score for 20 Newsgroups (K-Means): {news_purity_kmeans:.3f}\n")
    f.write(f"Purity Score for People Wikipedia (K-Means): {people_purity_kmeans:.3f}\n")
    
    f.write(f"Silhouette Score for People Wikipedia (Hierarchical): {silhouette_people_hierarc:.3f}\n")
    f.write(f"Silhouette Score for 20 Newsgroups (Hierarchical): {silhouette_news_hierarc:.3f}\n")
    f.write(f"Purity Score for 20 Newsgroups (Hierarchical): {news_purity_hierarc:.3f}\n")
    f.write(f"Purity Score for People Wikipedia (Hierarchical): {people_purity_hierarc:.3f}\n")

# Print confirmation
print("Results saved to results.txt!")

