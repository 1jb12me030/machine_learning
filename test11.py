import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv('sample_data.csv')

# Extract the data for t-SNE
matrix = df[['Age']].copy()

# Convert Age to a numeric column, replacing non-numeric values with NaN
matrix['Age'] = pd.to_numeric(matrix['Age'], errors='coerce')

# Drop rows with NaN values in the 'Age' column
matrix = matrix.dropna()

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

# Visualization code
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]
color_indices = range(len(matrix))

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.7)
plt.title("t-SNE Visualization of Age")
plt.show()
