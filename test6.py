# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Features
y = (X > 5).astype(int).flatten()  # Binary labels based on a threshold

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Plot the decision boundary
plt.scatter(X_test, y_test, color='black', label='True labels')
plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted labels')
plt.xlabel('Feature')
plt.ylabel('Class')
plt.legend()
plt.title('Logistic Regression - Binary Classification')
plt.show()
