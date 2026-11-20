# Experiment 23: Clustering Algorithms

This experiment demonstrates and compares three **unsupervised machine learning** clustering algorithms using a synthetic dataset:

1. **K-Means Clustering** – groups points around cluster centroids.
2. **Agglomerative Clustering** – builds clusters hierarchically by merging nearby groups.
3. **Gaussian Mixture Model (GMM)** – models clusters as a mixture of Gaussian distributions.

## Objective

Apply K-Means, Agglomerative Clustering, and Gaussian Mixture Model to the same dataset and visualize their results side by side.

## Requirements

- Python 3
- `scikit-learn`
- `matplotlib`

Install the required packages:

```bash
pip install scikit-learn matplotlib
```

## Python Code

Save the following as `clustering_algorithms.py`:

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate Synthetic Dataset
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
)

# Create 3 Subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1],
                c=kmeans_labels, cmap='viridis')
axes[0].set_title('K-Means Clustering')

# 2. Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=4)
agg_labels = agg.fit_predict(X)

axes[1].scatter(X[:, 0], X[:, 1],
                c=agg_labels, cmap='viridis')
axes[1].set_title('Agglomerative Clustering')

# 3. Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(X)

axes[2].scatter(X[:, 0], X[:, 1],
                c=gmm_labels, cmap='viridis')
axes[2].set_title('Gaussian Mixture Model')

# Display Graphs
plt.tight_layout()
plt.show()
```

## How to Run

1. Download or clone this repository.
2. Install the requirements using the command above.
3. Run the Python script:

   ```bash
   python clustering_algorithms.py
   ```

## Expected Output

A single figure with **three scatter plots**, one each for:

- K-Means Clustering
- Agglomerative Clustering
- Gaussian Mixture Model

Each plot displays the same 300 data points colored according to the clusters assigned by its algorithm. Note that cluster colors/label numbers can differ between algorithms even if their groupings are similar.

## Conclusion

All three algorithms can identify the four groups in this well-separated synthetic dataset. K-Means partitions by centroids, Agglomerative Clustering uses hierarchical merging, and GMM uses a probabilistic model.
