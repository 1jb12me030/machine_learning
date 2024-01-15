import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

df = pd.read_csv('sample_data.csv')

# Extract the data for t-SNE
matrix = df[['Age']].values

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

# Visualization code
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]
color_indices = range(len(df))

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.7)
plt.title("t-SNE Visualization of Age and City")
plt.show()

###########next################
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

#########next####################

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

####################next####################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Assuming your CSV file is named sample_data.csv
df = pd.read_csv('sample_data.csv')

# Encoding the 'City' column using one-hot encoding
encoder = OneHotEncoder(sparse_output=False)
city_encoded = encoder.fit_transform(df[['City']])

# Combining the one-hot encoded 'City' with the 'Age' column
X = pd.concat([df[['Age']], pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(['City']))], axis=1)
y = df['Age']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Printing the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#############next##########################


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Provided data
data = {
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Generate random values for Score and ada_embedding
df['Score'] = np.random.randint(1, 11, size=len(df))
df['ada_embedding'] = np.random.rand(len(df))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    list(df.ada_embedding.values),
    df.Score,
    test_size=0.2,
    random_state=42
)

# Combine X_train and y_train into a DataFrame for training data
train_data = pd.DataFrame({'ada_embedding': X_train, 'Score': y_train})

# Combine X_test and y_test into a DataFrame for testing data
test_data = pd.DataFrame({'ada_embedding': X_test, 'Score': y_test})

# Save the training and testing data to CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

# Display the first 20 rows of the training data
print(train_data.head(20))

################next##########################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Provided data
data = {
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Generate random values for Score and ada_embedding
df['Score'] = np.random.randint(1, 11, size=len(df))
df['ada_embedding'] = np.random.rand(len(df))

# Define features (X) and target variable (y)
X = df[['ada_embedding', 'Score']]
y = df['Name']  # Assuming 'Name' is the target variable for classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions on the test set
preds = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)

# Display the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)


###################next################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Provided data
data = {
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Generate random values for Score and ada_embedding
df['Score'] = np.random.randint(1, 11, size=len(df))
df['ada_embedding'] = np.random.rand(len(df))

# Define features (X) and target variable (y)
X = df[['ada_embedding', 'Score']]
y = df['Name']  # Assuming 'Name' is the target variable for classification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize and train the RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Make predictions on the test set
preds = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, preds)

# Adjust classification_report to handle undefined metrics
report = classification_report(y_test, preds, zero_division=1)

# Display the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

########################next###########################


import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = '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'

# Your data
data = {
    'Name': [ 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29],
    'City': ['Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas','x','y','z'],
}

# Creating DataFrame
df = pd.DataFrame(data)

# Generate embeddings for the 'City' column
embeddings = []
for city in df['City']:
    if city:
        response = openai.Completion.create(
            engine="text-embedding-ada-002",
            prompt=city,
            max_tokens=50,  # Set length to a value greater than 0
            n=1,
            logprobs=0,   # Specify logprobs parameter
            stop=None,
            temperature=0,
        )
        embedding = response['choices'][0]['text']
        embeddings.append(embedding)
    else:
        # Handle empty strings or missing values as needed
        embeddings.append(None)

# Ensure 'City' and 'City_Embeddings' have the same length
df['City_Embeddings'] = embeddings[:len(df['City'])]

# Display the DataFrame with embeddings
print(df[['Name', 'City', 'City_Embeddings']])


#########################next####################


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

####################next########################
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd

def search_reviews(df, product_description, n=3, pprint=True):
    try:
        embedding = get_embedding(product_description, model='text-embedding-ada-002')
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

# Sample data for demonstration purposes
df = pd.DataFrame({
    'ada_embedding': [get_embedding('tasty beans', model='text-embedding-ada-002'),
                      get_embedding('delicious beans', model='text-embedding-ada-002'),
                      get_embedding('awesome beans', model='text-embedding-ada-002')],
    'product_description': ['tasty beans', 'delicious beans', 'awesome beans']
})

df_people = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob', 'Eva', 'Daniel', 'Grace', 'Michael', 'Olivia', 'William', 'Sophia', 'Liam', 'Emily', 'Logan', 'Ava', 'Mason', 'Harper', 'Ethan', 'Emma', 'Oliver'],
    'Age': [25, 30, 22, 28, 35, 27, 32, 26, 29, 24, 31, 23, 34, 33, 28, 29, 26, 30, 27],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco', 'Seattle', 'Boston', 'Houston', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Dallas', 'Phoenix', 'Philadelphia', 'San Diego', 'Minneapolis', 'Portland', 'Detroit', 'Las Vegas']
})

# Searching for reviews and rearranging people data based on search results
res = search_reviews(df, 'delicious beans', n=3)

if res is not None:
    rearranged_people_data = df_people[df_people['Name'].isin(res.index)]
    print(rearranged_people_data)
else:
    print("Unable to get embeddings. Check your API key and network connectivity.")

