# Mall Customer Segmentation Using Clustering

A machine learning experiment comparing **K-Means**, **Agglomerative Clustering (Ward linkage)**, and **Gaussian Mixture Models (GMM)** for customer segmentation based on annual income and spending behavior.

The experiment creates **300 synthetic mall customers**, selects the number of clusters for each model, visualizes the resulting segments, and compares the assignments with the simulated groups.

## Objectives

- Group customers using their **annual income** and **spending score**.
- Compare three unsupervised clustering algorithms on the same data.
- Select cluster counts using silhouette scores or Bayesian Information Criterion (BIC).
- Visualize and interpret the customer segments.
- Evaluate agreement with the known synthetic groups using the Adjusted Rand Index (ARI).

## Dataset

The dataset is **generated in the notebook**, so no external CSV file is required.

| Feature | Description |
| --- | --- |
| `Annual income ($k)` | Annual income measured in thousands of US dollars |
| `Spending score` | Synthetic spending index from 1 (low) to 100 (high) |

The data contains 300 customers sampled from five simulated groups with different centers, spreads, and correlations. One middle-income group is elongated, allowing the experiment to compare how the algorithms handle different cluster shapes. A fixed random seed (`42`) makes the data generation reproducible.

> **Note:** These are simulated customers, not real mall transaction records.

## Algorithms

| Algorithm | Approach | Cluster-count selection |
| --- | --- | --- |
| **K-Means** | Assigns customers to the nearest centroid, minimizing within-cluster squared distances | Highest average silhouette score |
| **Agglomerative Clustering** | Uses Ward linkage to repeatedly merge clusters with the smallest increase in within-cluster variance | Highest average silhouette score |
| **Gaussian Mixture Model (GMM)** | Models customers as a mixture of Gaussian distributions with full covariance matrices | Lowest Bayesian Information Criterion (BIC) |

Each method is evaluated with **2 to 8 clusters** on standardized features.

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- IPython
- threadpoolctl
- Jupyter Notebook (to run the `.ipynb` version)

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib scipy scikit-learn ipython threadpoolctl notebook
```

## How to Run

### Run the Jupyter Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/Smith-005/ML.git
   cd ML
   ```

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook exp1.ipynb
   ```

3. Run the cells from top to bottom.

### Run the Python Script

If `exp1.py` is included in your repository, run:

```bash
python exp1.py
```

The script prints tables of model results and displays a three-panel clustering comparison chart.

## Experiment Workflow

1. **Generate data:** Draw 300 synthetic customers from five groups, then shuffle the rows.
2. **Preprocess:** Standardize annual income and spending score using `StandardScaler`.
3. **Fit models:** Train K-Means, Ward Agglomerative Clustering, and full-covariance GMM for cluster counts from 2 to 8.
4. **Choose cluster counts:** Use silhouette score for K-Means and Agglomerative Clustering; use BIC for GMM.
5. **Compare models:** Report silhouette scores and the ARI against the synthetic group labels.
6. **Visualize segments:** Plot the three clustering results in the original income and spending units.
7. **Interpret GMM:** Summarize segment sizes, average income, average spending score, and membership probabilities.
8. **Inspect uncertainty:** Find a customer whose GMM segment membership is most uncertain.

## Results

The notebook reports the following findings with the default random seed (`42`):

| Finding | Result |
| --- | --- |
| Selected number of clusters | **5 for all three methods** |
| Highest reported silhouette score | **K-Means: 0.618** |
| Highest agreement with simulated groups (ARI) | **GMM: 0.907** |

- **K-Means** yields compact, well-separated clusters according to the silhouette metric.
- **Agglomerative Clustering** identifies a similar broad segmentation, though some boundary customers receive different labels.
- **GMM** best matches the simulated groups according to ARI, and its full covariance matrices can model the elongated middle group.

The GMM-based profiles distinguish customers with similar income but different spending scores. The experiment finds lower-income/lower-spending, lower-income/higher-spending, higher-income/lower-spending, higher-income/higher-spending, and middle-range segments.

## Evaluation Metrics

**Silhouette score:** Measures separation between clusters and cohesion within them. Larger values indicate more distinct groupings.

**BIC (Bayesian Information Criterion):** Balances model fit and complexity for GMM. Smaller values are preferred.

**Adjusted Rand Index (ARI):** Measures agreement between predicted clusters and the known simulated groups, correcting for chance. Values near `1` indicate strong agreement. The simulated labels are **not used** for model fitting or cluster-count selection.

## Files

```text
ML/
├── README.md       # Project overview and instructions
├── exp1.ipynb      # Jupyter Notebook experiment
└── exp1.py         # Python script (if exported and added)
```

## Optional: Export Customer Assignments

The last notebook cell includes an optional CSV export. Uncomment this line to save the customer-level segment assignments:

```python
results.to_csv("mall_customer_segments.csv")
```

## Limitations

- The dataset is synthetic and designed with Gaussian populations; this naturally favors some of GMM's assumptions.
- The ARI is available only because the simulation provides known group labels. Real customer segmentation usually does not have these labels.
- Segments are illustrative and should not be treated as validated marketing categories without real customer data and domain analysis.

## Source

- [Original notebook: `exp1.ipynb`](https://github.com/Smith-005/ML/blob/main/exp1.ipynb)
