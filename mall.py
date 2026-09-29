import pandas as pd

df = pd.read_csv("Mall_Customers.csv")

print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print(X.head())

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

kmeans.fit(X)

df['Cluster'] = kmeans.labels_

print(df.head())


# -----------------------------------------
# STEP 4: Find Cluster Centroids
# -----------------------------------------

centroids = kmeans.cluster_centers_

print("\nCluster Centroids:")
print(centroids)


# -----------------------------------------
# STEP 5: Display Cluster Information
# -----------------------------------------

print("\nCustomer Count in Each Cluster:")
print(df['Cluster'].value_counts())


# -----------------------------------------
# STEP 6: Calculate Average Values
# -----------------------------------------

cluster_summary = df.groupby('Cluster')[
    ['Annual Income (k$)', 'Spending Score (1-100)']
].mean()

print("\nCluster Summary:")
print(cluster_summary)


# -----------------------------------------
# STEP 7: Visualize the Clusters
# -----------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df['Cluster'],
    s=80
)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=250,
    label='Centroids'
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.grid(True)

plt.show()


# -----------------------------------------
# STEP 8: Save Clustered Dataset
# -----------------------------------------

df.to_csv('Customer_Segmentation_Result.csv', index=False)

print("\nClustered dataset saved successfully!")


# -----------------------------------------
# STEP 9: Basic Interpretation
# -----------------------------------------

for cluster in sorted(df['Cluster'].unique()):

    income = cluster_summary.loc[cluster, 'Annual Income (k$)']
    spending = cluster_summary.loc[cluster, 'Spending Score (1-100)']
    customers = (df['Cluster'] == cluster).sum()

    print(f"\nCluster {cluster}")
    print(f"Customers: {customers}")
    print(f"Average Income: {income:.2f} k$")
    print(f"Average Spending Score: {spending:.2f}")