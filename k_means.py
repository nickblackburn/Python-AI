import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
path = "/Users/nicholas/Desktop/Machine Learning/data/Cust_Segmentation.csv"
cust_df = pd.read_csv(path)

# Drop the 'Address' column
df = cust_df.drop('Address', axis=1)

# Handle NaN values by imputing the mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df_imputed.values[:, 1:])

# Apply KMeans clustering
cluster_num = 3
k_means = KMeans(init="k-means++", n_clusters=cluster_num, n_init=12, random_state=42)
k_means.fit(X)
labels = k_means.labels_

# Add cluster labels to the DataFrame
df_imputed["Clus_km"] = labels

# Display cluster means
cluster_means = df_imputed.groupby('Clus_km').mean()
print(cluster_means)

# Scatter plot for Age vs. Income
plt.scatter(X[:, 0], X[:, 2], s=50, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

# 3D scatter plot for Age, Income, and Education
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 2], X[:, 1], c=labels.astype(float))

ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('Education')

plt.show()
