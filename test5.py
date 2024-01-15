
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming you have a labeled dataset
# X contains features, y contains labels (classes)
X, y = data_points_standardized, labels  # Replace 'labels' with your actual labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier (you can choose a different classifier)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
classifier.fit(X_train, y_train)

# Predict labels on the test set
y_pred = classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
