#Aditya Dutta & Ege Gürsel

# Perform RMF (Recency, Frequency, Monetary) transformation on customer data
# and segment customers into clusters using K-Means.

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_FILE = 'cleaned_online_retail.csv'
NUM_CLUSTERS = 5

print(f"Loading cleaned data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

# Parse invoice dates so we can compute recency for each customer.
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Define a snapshot date (one day after the latest invoice) to measure recency
# as the time since the customer's most recent purchase.
snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)

# Aggregate transactional data into RMF metrics per customer:
# Recency: days since last purchase
# Frequency: number of unique invoices
# Monetary: total spend.
rmf = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
})

rmf.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalAmount': 'Monetary'
}, inplace=True)

print(f"RMF profiles for {len(rmf)} customers created.")
print(rmf.head())

print("\nStandardizing RMF data...")

# Standardize RMF features so that each dimension has zero mean and unit variance,
# ensuring no single feature dominates the distance metric in K-Means.
scale = StandardScaler()
rmf_scaled = scale.fit_transform(rmf)

print(f"Running KMeans clustering with {NUM_CLUSTERS} clusters...")

# Fit a K-Means clustering model on the standardized RMF data.
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(rmf_scaled)

rmf['Cluster'] = kmeans.labels_

# Inspect average RMF values per cluster to understand the characteristics
# of each customer segment.
print("Cluster Averages:")
print(rmf.groupby('Cluster').mean())

print("\nVisualizing clusters...")

# Create a 3D scatter plot of customers in RMF space, colored by cluster label.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue', 'orange', 'purple']

for i in range(NUM_CLUSTERS):
    cluster_data = rmf[rmf['Cluster'] == i]
    ax.scatter(
        cluster_data['Recency'],
        cluster_data['Frequency'],
        cluster_data['Monetary'],
        c=colors[i],
        label=f'Cluster {i}',
        alpha=0.6,
        s = 50
    )

ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('Customer Segments based on RMF')
plt.legend()

plt.savefig('customer_clusters.png')
print("Cluster visualization saved to customer_clusters.png")

# Persist the RMF table with assigned clusters for further analysis or reporting.
rmf.to_csv('customer_rmf_clusters.csv')
print("RMF data with cluster labels saved to customer_rmf_clusters.csv")

print("Clustering completed.")
plt.show()