"""
TASK 3: Cluster Analysis - Run K-Means with Optimal Clusters and Analyze Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

os.makedirs('../outputs', exist_ok=True)

# Load data
df = pd.read_csv('../outputs/cleaned_customer_data.csv')

# Prepare features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load final cluster count
with open('../outputs/clustering_results.txt', 'r') as f:
    content = f.read()
    for line in content.split('\n'):
        if 'Final clusters to use:' in line:
            n_clusters = int(line.split(':')[1].strip())
            break
    else:
        n_clusters = 4

print(f"Running K-Means with {n_clusters} clusters...")

# Run K-Means with final number of clusters
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# Channel and Region mappings
channel_names = {
    1: "YouTube", 2: "Google", 3: "Instagram", 4: "LinkedIn",
    5: "Twitter/X", 6: "Facebook", 7: "Friend Referral", 8: "Other"
}
region_names = {0: "Rest of World", 1: "Europe", 2: "Anglo-Saxon"}

# Map to names for analysis
df['channel_name'] = df['channel'].map(channel_names)
df['region_name'] = df['region'].map(region_names)

# -------------------------------
# 1. Summary Table for Each Cluster
# -------------------------------

# Calculate summary statistics
cluster_summary = df.groupby('Cluster').agg({
    'CLV': ['mean', 'median', 'std'],
    'minutes_watched': ['mean', 'median', 'std'],
}).round(2)

# Add observation count and proportion
cluster_counts = df['Cluster'].value_counts().sort_index()
cluster_proportions = (cluster_counts / len(df) * 100).round(1)

cluster_summary['count'] = cluster_counts
cluster_summary['proportion_%'] = cluster_proportions

print("\n" + "=" * 70)
print("CLUSTER SUMMARY TABLE")
print("=" * 70)
print(cluster_summary)

# Save summary table
cluster_summary.to_csv('../outputs/cluster_summary.csv')
print("✓ Saved cluster_summary.csv")

# -------------------------------
# 2. Channel and Region Distribution by Cluster
# -------------------------------

# Channel distribution
channel_by_cluster = pd.crosstab(df['Cluster'], df['channel_name'], normalize='index') * 100
print("\n" + "=" * 70)
print("ACQUISITION CHANNEL DISTRIBUTION BY CLUSTER (%)")
print("=" * 70)
print(channel_by_cluster.round(1))
channel_by_cluster.to_csv('../outputs/channel_by_cluster.csv')
print("✓ Saved channel_by_cluster.csv")

# Region distribution
region_by_cluster = pd.crosstab(df['Cluster'], df['region_name'], normalize='index') * 100
print("\n" + "=" * 70)
print("COUNTRY REGION DISTRIBUTION BY CLUSTER (%)")
print("=" * 70)
print(region_by_cluster.round(1))
region_by_cluster.to_csv('../outputs/region_by_cluster.csv')
print("✓ Saved region_by_cluster.csv")

# -------------------------------
# 3. Name and Explain Each Cluster
# -------------------------------

print("\n" + "=" * 70)
print("CLUSTER NAMES AND EXPLANATIONS")
print("=" * 70)

cluster_profiles = []

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    # Calculate key metrics
    avg_clv = cluster_data['CLV'].mean()
    avg_minutes = cluster_data['minutes_watched'].mean()
    size_pct = len(cluster_data) / len(df) * 100
    
    # Most common channel and region
    top_channel = cluster_data['channel_name'].mode()[0] if not cluster_data['channel_name'].mode().empty else "Unknown"
    top_region = cluster_data['region_name'].mode()[0] if not cluster_data['region_name'].mode().empty else "Unknown"
    
    # Engagement level
    if avg_minutes > df['minutes_watched'].quantile(0.75):
        engagement = "High"
    elif avg_minutes < df['minutes_watched'].quantile(0.25):
        engagement = "Low"
    else:
        engagement = "Medium"
    
    # CLV level
    if avg_clv > df['CLV'].quantile(0.75):
        clv_level = "High"
    elif avg_clv < df['CLV'].quantile(0.25):
        clv_level = "Low"
    else:
        clv_level = "Medium"
    
    # Generate cluster name based on characteristics
    if avg_minutes > 100000:
        name = "🎯 Ultra-Engaged Power Users"
        explanation = f"Extremely high engagement ({avg_minutes:.0f} minutes) with premium CLV (${avg_clv:.0f})"
    elif engagement == "High" and clv_level == "High":
        name = "⭐ Elite Engaged Members"
        explanation = f"Highly engaged students with premium CLV, primarily from {top_region} via {top_channel}"
    elif engagement == "High" and clv_level == "Medium":
        name = "📚 Active Learners"
        explanation = f"Engaged students with moderate CLV, primarily from {top_region} via {top_channel}"
    elif engagement == "Medium" and clv_level == "Medium":
        name = "📖 Regular Users"
        explanation = f"Moderate engagement and CLV, primarily from {top_region} via {top_channel}"
    elif clv_level == "Low":
        name = "💤 Low-Value Users"
        explanation = f"Lower CLV, needs engagement boost, primarily from {top_region} via {top_channel}"
    else:
        name = f"Cluster {cluster_id}"
        explanation = f"Mixed characteristics, {size_pct:.1f}% of customers"
    
    cluster_profiles.append({
        'Cluster': cluster_id,
        'name': name,
        'explanation': explanation,
        'size_pct': round(size_pct, 1),
        'avg_clv': round(avg_clv, 2),
        'avg_minutes': round(avg_minutes, 0),
        'top_channel': top_channel,
        'top_region': top_region,
        'engagement': engagement,
        'clv_level': clv_level
    })
    
    print(f"\nCluster {cluster_id}: {name}")
    print(f"  → {explanation}")
    print(f"  → Size: {size_pct:.1f}% of customers")
    print(f"  → Avg CLV: ${avg_clv:.2f}")
    print(f"  → Avg Minutes Watched: {avg_minutes:.0f}")
    print(f"  → Top Channel: {top_channel}")
    print(f"  → Top Region: {top_region}")

# Save cluster profiles
cluster_profiles_df = pd.DataFrame(cluster_profiles)
cluster_profiles_df.to_csv('../outputs/cluster_profiles.csv', index=False)
print("\n✓ Saved cluster_profiles.csv")

# -------------------------------
# 4. Visualizations
# -------------------------------

# 1. Cluster distribution pie chart
plt.figure(figsize=(10, 6))
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
plt.pie(cluster_counts.values, labels=[f"{c}: {cluster_profiles[c]['name'][:20]}" for c in range(n_clusters)], 
        autopct='%1.1f%%', colors=colors[:n_clusters], startangle=90)
plt.title('Customer Distribution by Cluster')
plt.tight_layout()
plt.savefig('../outputs/cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cluster_distribution.png")

# 2. CLV vs Minutes Watched by Cluster
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['CLV'], df['minutes_watched'], c=df['Cluster'], cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Customer Lifetime Value (CLV)')
plt.ylabel('Minutes Watched')
plt.title('CLV vs Minutes Watched Colored by Cluster')
plt.tight_layout()
plt.savefig('../outputs/cluster_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cluster_scatter.png")

# 3. Box plots by cluster
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df.boxplot(column='CLV', by='Cluster', ax=axes[0])
axes[0].set_title('CLV Distribution by Cluster')
axes[0].set_ylabel('CLV')
axes[0].set_xlabel('Cluster')

df.boxplot(column='minutes_watched', by='Cluster', ax=axes[1])
axes[1].set_title('Minutes Watched Distribution by Cluster')
axes[1].set_ylabel('Minutes Watched')
axes[1].set_xlabel('Cluster')

plt.suptitle('')
plt.tight_layout()
plt.savefig('../outputs/cluster_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved cluster_boxplots.png")

print("\n" + "=" * 50)
print("TASK 3 COMPLETE: Cluster analysis and profiling done")
print("=" * 50)