import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page setup
st.set_page_config(page_title="K-Means Clustering App", layout="wide")
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Sidebar - choose k
st.sidebar.header("Configure Clustering")
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, value=3)

# Load Iris dataset
iris = load_iris()
X = iris.data

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Define fixed color palette
colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink', 'brown', 'gray', 'olive']

# Plot
fig, ax = plt.subplots()
for i in range(n_clusters):
    ax.scatter(
        X_pca[y_kmeans == i, 0],
        X_pca[y_kmeans == i, 1],
        s=50,
        c=colors[i],
        label=f'Cluster {i}'
    )

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Centroids')

ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()
st.pyplot(fig)
