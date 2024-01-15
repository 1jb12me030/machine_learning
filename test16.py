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
