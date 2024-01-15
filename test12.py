import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('sample_data.csv')

# Extract the data for t-SNE
matrix = df[['Name', 'City']].copy()

# Drop rows with NaN values in the 'Name' or 'City' columns
matrix = matrix.dropna(subset=['Name', 'City'])

# Label encode 'Name' and 'City' columns
label_encoder = LabelEncoder()
matrix['Name'] = label_encoder.fit_transform(matrix['Name'])
matrix['City'] = label_encoder.fit_transform(matrix['City'])

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
plt.title("t-SNE Visualization of Name and City")
plt.show()
