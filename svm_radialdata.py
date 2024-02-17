# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score


# Generate radial data
X, y = make_circles(n_samples=100, noise=0.1, factor=0.4, random_state=42)

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Radially Distributed Data')
plt.show()

# Create and train the SVM model
svm_model = SVC(kernel='rbf', gamma='auto', random_state=42)
svm_model.fit(X, y)

# Predict labels for the data
y_pred = svm_model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Recall:", recall)




# Plot the decision boundary
def plot_decision_boundary(model, X, y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot decision boundary and data points
plt.figure(figsize=(8, 6))
plot_decision_boundary(svm_model, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary on Radial Data')
plt.show()



# Generate more complex radial data with closely intermingled red and blue points
X, y = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Complex Radially Distributed Data')
plt.show()

# Create and train the SVM model
svm_model = SVC(kernel='rbf', gamma='auto', random_state=42)
svm_model.fit(X, y)

# Predict labels for the data
y_pred = svm_model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Recall:", recall)


# Plot decision boundary and data points
plt.figure(figsize=(8, 6))
plot_decision_boundary(svm_model, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary on Complex Radial Data')
plt.show()

# Generate paired data points
np.random.seed(42)
num_points = 100
blue_points = np.random.rand(num_points, 2)
red_points = blue_points + 0.1 * np.random.randn(num_points, 2)  # Add noise to create slight variation

# Combine blue and red points
X = np.vstack((blue_points, red_points))
y = np.hstack((np.zeros(num_points), np.ones(num_points)))

# Shuffle the data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Paired Data Points (Blue and Red)')
plt.show()

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X, y)

# Predict labels for the data
y_pred = svm_model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Recall:", recall)


# Plot decision boundary and data points
plt.figure(figsize=(8, 6))
plot_decision_boundary(svm_model, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary on Paired Data')
plt.show()



# Generate paired data points
np.random.seed(42)
num_points = 100
blue_points = np.random.rand(num_points, 2)
red_points = blue_points + 0.1 * np.random.randn(num_points, 2)  # Add noise to create slight variation

# Combine blue and red points
X = np.vstack((blue_points, red_points))
y = np.hstack((np.zeros(num_points), np.ones(num_points)))

# Shuffle the data
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X, y)

# Predict labels for the data
y_pred = svm_model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Recall:", recall)

# Plot decision boundary and data points
plt.figure(figsize=(8, 6))
plot_decision_boundary(svm_model, X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary on Paired Data')
plt.show()