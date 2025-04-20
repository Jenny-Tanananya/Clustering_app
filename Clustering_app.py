import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="K-Means Clustering App", layout="wide")

# Title
st.title("🔍 K-Means Clustering App with Iris Dataset by Tanananya Thongkum")

# Sidebar
st.sidebar.header("Configure Clustering")
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, value=3)

# Load Iris dataset
iris = load_iris()
X = iris.data
df = pd.DataFrame(X, columns=iris.feature_names)

# Apply KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# PCA for 2D projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='rainbow', s=50)
ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

# Add legend
for i in range(n_clusters):
    ax.scatter([], [], c=scatter.cmap(i / n_clusters), label=f'Cluster {i}')
ax.legend()

# Display plot
st.pyplot(fig)
