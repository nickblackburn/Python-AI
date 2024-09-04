import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Read the CSV file
path = "/Users/nicholas/Desktop/Machine Learning/data/teleCust1000t.csv"
df = pd.read_csv(path)

# Feature selection
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Standardize the features
scaler = preprocessing.StandardScaler()
X_train_norm = scaler.fit_transform(X_train.astype(float))
X_test_norm = scaler.transform(X_test.astype(float))

def train_and_evaluate_model(k, X_train, y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    
    train_accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, yhat)
    
    return train_accuracy, test_accuracy

# Find the best k
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    train_acc, test_acc = train_and_evaluate_model(n, X_train_norm, y_train, X_test_norm, y_test)
    mean_acc[n-1] = test_acc
    std_acc[n-1] = np.std(train_acc)

# Plot the results
plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

best_k = np.argmax(mean_acc) + 1
print(f"The best accuracy was with {mean_acc.max()} with k={best_k}")
