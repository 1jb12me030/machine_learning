
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Provided vector data
vector_data_ada_002 = [
    0.002253932, -0.009333183, 0.01574578, -0.007790351, -0.004711035, 0.014844206, -0.009739526, -0.03822161,
    # ... (remaining elements)

    0.020609193, 0.020507608, -0.0062633916, -0.0052475347, 0.009199852, 0.013472799, -0.01438707, 0.0035618476,
    -0.011206169, -0.018018758, -0.0152251525, -0.013739462, 0.023644064
]

# Generating synthetic binary labels (0 or 1) for demonstration purposes
y = (np.random.rand(100) > 0.5).astype(int)

# Reshape the vector_data to make it compatible with scikit-learn
X = np.array(vector_data_ada_002).reshape(-1, 1)

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
plt.title('Logistic Regression - Binary Classification with Provided Data')
plt.show()
