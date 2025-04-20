import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")
st.title("🔍 K-Means Clustering App with Iris Dataset by Tanananya Thongkum")

# Sidebar for cluster selection
st.sidebar.header("Clustering Options")
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, value=3)

# Load dataset
iris = load_iris()
X = iris.data

# Fit model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Define custom color palette
color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'pink', 'brown', 'gray', 'olive']
cluster_colors = [color_palette[label] for label in clusters]

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors, s=50)
ax.set_title("Clusters (PCA Visualization)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Legend manually
for i in range(n_clusters):
    ax.scatter([], [], c=color_palette[i], label=f"Cluster {i}")
ax.legend()

# Show plot
st.pyplot(fig)
