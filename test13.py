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
