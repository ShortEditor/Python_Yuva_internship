# Week 3: Unsupervised Learning and Clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
X = iris.data
X.columns = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compare candidate values of K
scores = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    scores.append((k, model.inertia_, silhouette_score(X_scaled, labels)))

metrics = pd.DataFrame(scores, columns=["K","Inertia","Silhouette"])
print(metrics)

# Final K=3 solution: selected from elbow/interpretability trade-off.
# Silhouette alone favors K=2; this is reported explicitly in the report.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

result = X.copy()
result["Cluster"] = labels
print(result.groupby("Cluster").mean())

# PCA only for 2-D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Secondary hierarchical comparison
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
agg_labels = agg.fit_predict(X_scaled)
print("Hierarchical silhouette:", silhouette_score(X_scaled, agg_labels))

for c in sorted(result["Cluster"].unique()):
    idx = result["Cluster"] == c
    plt.scatter(X_pca[idx,0], X_pca[idx,1], label=f"Cluster {c}")
plt.title("K-Means Clusters Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
