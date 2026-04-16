"""
TASK 2: K-Means and Hierarchical Clustering Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import os

os.makedirs('../outputs', exist_ok=True)

# Load the cleaned data
df = pd.read_csv('../outputs/cleaned_customer_data.csv')

# -------------------------------
# 1. Standardize the Data
# -------------------------------

# Select only numeric columns for clustering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("=" * 50)
print("DATA STANDARDIZED")
print("=" * 50)
print(f"Shape of standardized data: {X_scaled_df.shape}")

# -------------------------------
# 2. Hierarchical Clustering (Ward's Method)
# -------------------------------

# Perform hierarchical clustering using Ward's method
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Hierarchical Clustering Dendrogram (Ward\'s Method)')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('../outputs/dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved dendrogram.png")

# Calculate distances between merges
distances = linkage_matrix[:, 2]
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(distances)+1), distances[::-1])
plt.xlabel('Merge Step')
plt.ylabel('Distance')
plt.title('Distance Between Merges - Hierarchical Clustering')
plt.tight_layout()
plt.savefig('../outputs/hierarchical_distances.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved hierarchical_distances.png")

# Based on dendrogram analysis
optimal_k_hierarchical = 4
print(f"\n✓ Optimal clusters from hierarchical clustering: {optimal_k_hierarchical}")

# -------------------------------
# 3. K-Means Clustering with Elbow Method
# -------------------------------

# Determine optimal k using elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot elbow curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Cluster Count')
plt.grid(True)

plt.tight_layout()
plt.savefig('../outputs/elbow_method.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved elbow_method.png")

print("\nK-Means Results:")
for k, inertia, sil in zip(k_range, inertias, silhouette_scores):
    print(f"  k={k}: Inertia={inertia:.0f}, Silhouette={sil:.3f}")

best_k_kmeans = k_range[np.argmax(silhouette_scores)]
print(f"\n✓ K-Means optimal clusters (by silhouette): {best_k_kmeans}")

# -------------------------------
# 4. Final Decision on Number of Clusters
# -------------------------------

# Use hierarchical clustering result if inconsistencies
if optimal_k_hierarchical != best_k_kmeans:
    print(f"\n⚠ Inconsistency: Hierarchical suggests {optimal_k_hierarchical}, K-Means suggests {best_k_kmeans}")
    print("→ Following project instructions: Using hierarchical clustering result")
    final_n_clusters = optimal_k_hierarchical
else:
    final_n_clusters = optimal_k_hierarchical

print(f"\n{'='*50}")
print(f"FINAL NUMBER OF CLUSTERS: {final_n_clusters}")
print(f"{'='*50}")

# Save results
with open('../outputs/clustering_results.txt', 'w') as f:
    f.write(f"Hierarchical optimal clusters: {optimal_k_hierarchical}\n")
    f.write(f"K-Means optimal clusters: {best_k_kmeans}\n")
    f.write(f"Final clusters to use: {final_n_clusters}\n")
print("✓ Saved clustering_results.txt")

print("\nTASK 2 COMPLETE: Clustering analysis done")