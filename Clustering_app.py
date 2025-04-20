import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

#set the page config
st.set_page_config(page_title="K-Means Clustering App", layout="centered")

#set title
st.title("K-Means Clustering Visualizer by Tanananya Thongkum")

# Sidebar for selecting k
st.sidebar.title("Configure Clustering")
n_clusters = st.sidebar.slider("Select number of clusters (k)", 2, 10, value=3)

#display Dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predict using the loaded model
model = KMeans(n_clusters=n_clusters, random_state=42)
y_kmeans = loaded_model.predict(X)

#plotting 
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
