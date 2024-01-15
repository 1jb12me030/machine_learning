
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Assuming df is your DataFrame with Name, Age, City, and ada_embedding columns

# Example data
data = {'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
        'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
        'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas'],
        'ada_embedding': [np.random.rand(10) for _ in range(19)]}

df = pd.DataFrame(data)

# Stack ada_embedding values into a matrix
matrix = np.vstack(df.ada_embedding.values)

# Set the number of clusters
n_clusters = 4

# Initialize KMeans model and fit the data
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)

# Add a new column 'Cluster' to the DataFrame indicating the cluster for each row
df['Cluster'] = kmeans.labels_

# Display the DataFrame with the assigned clusters
print(df)
import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(10, 6))

# Scatter points for each cluster
for cluster_label in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_label]
    plt.scatter(cluster_data['Age'], cluster_data['ada_embedding'].apply(lambda x: x[0]), label=f'Cluster {cluster_label}')

# Plot settings
plt.title('K-Means Clustering Visualization')
plt.xlabel('Age')
plt.ylabel('Ada Embedding (First Dimension)')
plt.legend()
plt.grid(True)
plt.show()
