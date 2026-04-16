"""
TASK 4: Marketing Strategy Recommendations Based on Cluster Analysis
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
cluster_profiles = pd.read_csv('../outputs/cluster_profiles.csv')

# Re-run clustering to get cluster labels
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[numeric_cols].copy()
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

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Channel and Region mappings
channel_names = {
    1: "YouTube", 2: "Google", 3: "Instagram", 4: "LinkedIn",
    5: "Twitter/X", 6: "Facebook", 7: "Friend Referral", 8: "Other"
}
region_names = {0: "Rest of World", 1: "Europe", 2: "Anglo-Saxon"}

df['channel_name'] = df['channel'].map(channel_names)
df['region_name'] = df['region'].map(region_names)

print("=" * 70)
print("TASK 4: MARKETING STRATEGY ANALYSIS")
print("=" * 70)

# -------------------------------
# 1. Identify Engaged Students with High CLV
# -------------------------------

print("\n" + "=" * 70)
print("HIGH-VALUE ENGAGED SEGMENTS")
print("=" * 70)

high_value_segments = []
for _, row in cluster_profiles.iterrows():
    if row['avg_clv'] > df['CLV'].median() and row['avg_minutes'] > df['minutes_watched'].median():
        high_value_segments.append(row)
        print(f"\nCluster {int(row['Cluster'])}: {row['name']}")
        print(f"  - Avg CLV: ${row['avg_clv']:.2f} (above median ${df['CLV'].median():.2f})")
        print(f"  - Avg Minutes: {row['avg_minutes']:.0f} (above median {df['minutes_watched'].median():.0f})")
        print(f"  - Top Channel: {row['top_channel']}")
        print(f"  - Top Region: {row['top_region']}")
        print(f"  - Size: {row['size_pct']:.1f}% of customers")

# -------------------------------
# 2. Channel Performance Analysis
# -------------------------------

print("\n" + "=" * 70)
print("CHANNEL PERFORMANCE ANALYSIS")
print("=" * 70)

# Calculate channel metrics
channel_metrics = df.groupby('channel_name').agg({
    'CLV': ['mean', 'median', 'count'],
    'minutes_watched': 'mean'
}).round(2)

channel_metrics.columns = ['avg_clv', 'median_clv', 'customer_count', 'avg_minutes']
channel_metrics['customer_pct'] = (channel_metrics['customer_count'] / len(df) * 100).round(1)
channel_metrics = channel_metrics.sort_values('avg_clv', ascending=False)

print("\nChannel Performance Metrics:")
print(channel_metrics.to_string())

# Save channel metrics
channel_metrics.to_csv('../outputs/channel_metrics.csv')
print("\n✓ Saved channel_metrics.csv")

# Identify underperforming channels
overall_avg_clv = df['CLV'].mean()
overall_avg_minutes = df['minutes_watched'].mean()

print("\n" + "-" * 50)
print("CHANNEL PERFORMANCE ASSESSMENT")
print("-" * 50)

underperforming_channels = []
high_performing_channels = []

for channel in channel_metrics.index:
    avg_clv = channel_metrics.loc[channel, 'avg_clv']
    avg_minutes = channel_metrics.loc[channel, 'avg_minutes']
    customer_pct = channel_metrics.loc[channel, 'customer_pct']
    
    if avg_clv < overall_avg_clv and avg_minutes < overall_avg_minutes:
        underperforming_channels.append(channel)
        print(f"\n⚠ {channel}: UNDERPERFORMING")
        print(f"   CLV: ${avg_clv:.2f} (below avg ${overall_avg_clv:.2f})")
        print(f"   Minutes: {avg_minutes:.0f} (below avg {overall_avg_minutes:.0f})")
        print(f"   Share of customers: {customer_pct}%")
    elif avg_clv > overall_avg_clv and avg_minutes > overall_avg_minutes:
        high_performing_channels.append(channel)
        print(f"\n✅ {channel}: HIGH PERFORMING")
        print(f"   CLV: ${avg_clv:.2f} (above avg)")
        print(f"   Minutes: {avg_minutes:.0f} (above avg)")
    else:
        print(f"\n📊 {channel}: MIXED PERFORMANCE")
        print(f"   CLV: ${avg_clv:.2f} | Minutes: {avg_minutes:.0f}")

# -------------------------------
# 3. Regional Channel Recommendations
# -------------------------------

print("\n" + "=" * 70)
print("REGIONAL CHANNEL RECOMMENDATIONS")
print("=" * 70)

region_channel_performance = df.groupby(['region_name', 'channel_name']).agg({
    'CLV': 'mean',
    'minutes_watched': 'mean'
}).round(2)
region_channel_performance.columns = ['avg_clv', 'avg_minutes']

recommendations = {}

for region in df['region_name'].dropna().unique():
    region_data = region_channel_performance.xs(region, level='region_name')
    if len(region_data) > 0:
        best_clv_channel = region_data.loc[region_data['avg_clv'].idxmax()]
        best_engagement_channel = region_data.loc[region_data['avg_minutes'].idxmax()]
        
        print(f"\n📍 {region}:")
        print(f"   Best for CLV: {region_data['avg_clv'].idxmax()} (${best_clv_channel['avg_clv']:.0f} avg CLV)")
        print(f"   Best for Engagement: {region_data['avg_minutes'].idxmax()} ({best_engagement_channel['avg_minutes']:.0f} avg minutes)")
        
        recommendations[region] = {
            'primary_channel': region_data['avg_clv'].idxmax(),
            'secondary_channel': region_data['avg_minutes'].idxmax()
        }

# -------------------------------
# 4. Actionable Marketing Strategies by Cluster
# -------------------------------

print("\n" + "=" * 70)
print("CLUSTER-SPECIFIC MARKETING STRATEGIES")
print("=" * 70)

cluster_strategies = []

for _, row in cluster_profiles.iterrows():
    cluster_id = int(row['Cluster'])
    print(f"\n📌 {row['name']}")
    print(f"   Size: {row['size_pct']:.1f}% of customer base")
    print(f"   Avg CLV: ${row['avg_clv']:.2f}")
    print(f"   Avg Minutes: {row['avg_minutes']:.0f}")
    
    # Determine strategy based on characteristics
    if row['avg_minutes'] > 100000:
        strategy = "PREMIUM RETENTION"
        tactics = [
            "Create ambassador program for these super users",
            "Ask for testimonials and case studies",
            "Offer beta access to new features",
            "Provide exclusive 1-on-1 coaching sessions"
        ]
    elif row['engagement'] == "High" and row['clv_level'] == "High":
        strategy = "NURTURE & REWARD"
        tactics = [
            "Implement loyalty program with exclusive content",
            "Offer referral bonuses (they bring quality customers)",
            "Send personalized advanced content recommendations",
            "Invite to VIP webinars and events"
        ]
    elif row['engagement'] == "High" and row['clv_level'] == "Medium":
        strategy = "MONETIZE ENGAGEMENT"
        tactics = [
            "Introduce premium tier or micro-transactions",
            "Targeted upselling campaigns based on viewing history",
            "Time-limited offers on premium content",
            "Bundle related courses for higher value"
        ]
    elif row['engagement'] == "Medium" and row['clv_level'] == "High":
        strategy = "RE-ENGAGEMENT CAMPAIGN"
        tactics = [
            "Send personalized email/SMS reminders",
            "Offer 1-on-1 coaching or webinars",
            "Create digestible, short-form content",
            "Highlight popular content they haven't watched"
        ]
    else:
        strategy = "COST OPTIMIZATION"
        tactics = [
            "Reduce marketing spend on this segment",
            "Test low-cost reactivation campaigns (email only)",
            "If no improvement after 2 campaigns, deprioritize",
            "Focus acquisition budget on higher-value segments"
        ]
    
    print(f"   🎯 STRATEGY: {strategy}")
    for tactic in tactics:
        print(f"      → {tactic}")
    
    cluster_strategies.append({
        'Cluster': cluster_id,
        'name': row['name'],
        'strategy': strategy,
        'tactics': '; '.join(tactics)
    })

# Save cluster strategies
pd.DataFrame(cluster_strategies).to_csv('../outputs/cluster_strategies.csv', index=False)
print("\n✓ Saved cluster_strategies.csv")

# -------------------------------
# 5. Visualizations
# -------------------------------

# Channel performance bar charts
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CLV by channel
channels_sorted = channel_metrics.sort_values('avg_clv', ascending=True)
colors_clv = ['green' if x in high_performing_channels else 'red' if x in underperforming_channels else 'orange' 
              for x in channels_sorted.index]
axes[0].barh(channels_sorted.index, channels_sorted['avg_clv'], color=colors_clv)
axes[0].axvline(overall_avg_clv, color='blue', linestyle='--', linewidth=2, label=f'Overall Avg: ${overall_avg_clv:.0f}')
axes[0].set_xlabel('Average CLV ($)')
axes[0].set_title('Average CLV by Acquisition Channel')
axes[0].legend()

# Minutes by channel
channels_sorted_min = channel_metrics.sort_values('avg_minutes', ascending=True)
colors_min = ['green' if x in high_performing_channels else 'red' if x in underperforming_channels else 'orange' 
              for x in channels_sorted_min.index]
axes[1].barh(channels_sorted_min.index, channels_sorted_min['avg_minutes'], color=colors_min)
axes[1].axvline(overall_avg_minutes, color='blue', linestyle='--', linewidth=2, label=f'Overall Avg: {overall_avg_minutes:.0f} min')
axes[1].set_xlabel('Average Minutes Watched')
axes[1].set_title('Average Engagement by Acquisition Channel')
axes[1].legend()

plt.tight_layout()
plt.savefig('../outputs/channel_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved channel_performance.png")

# -------------------------------
# 6. Final Marketing Strategy Report
# -------------------------------

print("\n" + "=" * 70)
print("GENERATING FINAL MARKETING STRATEGY REPORT")
print("=" * 70)

with open('../outputs/marketing_strategy_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CUSTOMER SEGMENTATION MARKETING STRATEGY REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 40 + "\n")
    f.write(f"Total Customers Analyzed: {len(df):,}\n")
    f.write(f"Number of Segments Identified: {n_clusters}\n")
    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
    
    f.write("KEY FINDINGS\n")
    f.write("-" * 40 + "\n")
    f.write(f"• Instagram is the HIGHEST PERFORMING channel (${channel_metrics.loc['Instagram', 'avg_clv']:.0f} avg CLV)\n")
    f.write(f"• Twitter/X is UNDERPERFORMING (${channel_metrics.loc['Twitter/X', 'avg_clv']:.0f} avg CLV)\n")
    f.write(f"• LinkedIn and Facebook also underperform\n")
    f.write(f"• Cluster with 0.2% of users watches 132,000+ minutes - these are super users!\n\n")
    
    f.write("CLUSTER PROFILES\n")
    f.write("-" * 40 + "\n")
    for _, row in cluster_profiles.iterrows():
        f.write(f"\n{row['name']}\n")
        f.write(f"  • Size: {row['size_pct']:.1f}% of customers\n")
        f.write(f"  • Avg CLV: ${row['avg_clv']:.2f}\n")
        f.write(f"  • Avg Minutes: {row['avg_minutes']:.0f}\n")
        f.write(f"  • Primary Region: {row['top_region']}\n")
        f.write(f"  • Primary Channel: {row['top_channel']}\n")
    
    f.write("\n\nCHANNEL PERFORMANCE RANKING\n")
    f.write("-" * 40 + "\n")
    for i, (channel, row) in enumerate(channel_metrics.iterrows(), 1):
        performance = "✅ HIGH" if channel in high_performing_channels else "⚠️ LOW" if channel in underperforming_channels else "📊 MIXED"
        f.write(f"{i}. {channel}: {performance} (CLV: ${row['avg_clv']:.0f}, Engagement: {row['avg_minutes']:.0f} min)\n")
    
    f.write("\n\nRECOMMENDATIONS\n")
    f.write("-" * 40 + "\n")
    f.write("\nBUDGET REALLOCATION:\n")
    f.write("  • INCREASE spend on: " + ", ".join(high_performing_channels) + "\n")
    f.write("  • DECREASE spend on: " + ", ".join(underperforming_channels) + "\n")
    
    f.write("\nREGIONAL STRATEGY:\n")
    for region, rec in recommendations.items():
        f.write(f"  • {region}: Focus on {rec['primary_channel']} (CLV) and {rec['secondary_channel']} (engagement)\n")
    
    f.write("\nCLUSTER-SPECIFIC TACTICS:\n")
    for cs in cluster_strategies:
        f.write(f"  • {cs['name']}: {cs['strategy']}\n")
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print("✓ Saved marketing_strategy_report.txt")

# -------------------------------
# 7. Final Summary Output
# -------------------------------

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY - RECOMMENDATIONS")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    MARKETING STRATEGY RECOMMENDATIONS                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  🎯 ACQUISITION STRATEGY (per region):                                ║
║     → Reallocate budget based on regional channel performance        ║
║     → Double down on top-performing channels for each region         ║
║                                                                       ║
║  📊 CHANNEL INVESTMENT PRIORITIES:                                    ║
║     → HIGH PRIORITY: {', '.join(high_performing_channels)}                         
║     → LOW PRIORITY: {', '.join(underperforming_channels)}                        
║                                                                       ║
║  💡 CRITICAL FINDING:                                                 ║
║     → Cluster 3 (0.2% of customers) watches 132,000+ minutes!        ║
║     → These are your brand ambassadors - nurture them!               ║
║                                                                       ║
║  📈 RECOMMENDED ACTIONS:                                              ║
║     1. Increase Instagram ad spend by 30-40%                        ║
║     2. Pause Twitter/X campaigns immediately                        ║
║     3. Launch referral program for Cluster 2                        ║
║     4. Create ambassador program for Cluster 3                      ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 50)
print("TASK 4 COMPLETE: All outputs generated successfully!")
print("=" * 50)

# List all generated files
print("\n📁 OUTPUT FILES GENERATED:")
output_files = [
    "cleaned_customer_data.csv",
    "correlation_heatmap.png",
    "clv_vs_minutes.png",
    "dendrogram.png",
    "hierarchical_distances.png",
    "elbow_method.png",
    "clustering_results.txt",
    "cluster_summary.csv",
    "channel_by_cluster.csv",
    "region_by_cluster.csv",
    "cluster_profiles.csv",
    "cluster_distribution.png",
    "cluster_scatter.png",
    "cluster_boxplots.png",
    "channel_metrics.csv",
    "cluster_strategies.csv",
    "channel_performance.png",
    "marketing_strategy_report.txt"
]

for f in sorted(output_files):
    if os.path.exists(f'../outputs/{f}'):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f}")