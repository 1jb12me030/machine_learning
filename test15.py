
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
