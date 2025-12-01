import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

INPUT_FILE = 'cleaned_online_retail.csv'
NUM_CLUSTERS = 5

print(f"Loading cleaned data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

print("Tranformaing data into RMF view...")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

snapshot_date = df['InvoiceDate'].max() + pd.DateOffset(days=1)

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

scale = StandardScaler()
rmf_scaled = scale.fit_transform(rmf)

print(f"Running KMeans clustering with {NUM_CLUSTERS} clusters...")

kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
kmeans.fit(rmf_scaled)

rmf['Cluster'] = kmeans.labels_

print("Cluster Averages:")
print(rmf.groupby('Cluster').mean())

print("\nVisualizing clusters...")
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

rmf.to_csv('customer_rmf_clusters.csv')
print("RMF data with cluster labels saved to customer_rmf_clusters.csv")

print("Clustering completed.")
plt.show()