import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt

def calculate_distance(obj1, obj2):
    """Calculate Euclidean distance between two objects."""
    return np.sqrt(np.sum((obj1 - obj2) ** 2))

def kmeans_tensorflow(k, D, attributes):
    # Step 1: Choose k random elements from D as initial cluster centers
    centroids = np.array(random.sample(D, k))
    print(centroids)

    while True:
        # Step 2: Assign each object to the cluster based on the distance to the cluster center
        distances = cdist(np.array([obj[attributes] for obj in D]), centroids[:, attributes])
        assigned_clusters = np.argmin(distances, axis=1)

        # Step 3: Update the center of each cluster based on the new composition of the cluster
        new_centroids = np.array([np.mean(np.array([D[i][attributes] for i in range(len(D)) if assigned_clusters[i] == j]), axis=0) for j in range(k)])
        print(new_centroids)

        # Step 4: Check for convergence (if centroids do not change)
        if np.array_equal(new_centroids[:, attributes], centroids[:, attributes]):
            break
        centroids = new_centroids

    return centroids

if __name__ == "__main__":
    # Generate sample data
    data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

    # Convert data to a Pandas DataFrame
    df = pd.DataFrame(data, columns=['X1', 'X2'])

    # Using the alternative k-means implementation with TensorFlow and Keras
    kmeans_tensorflow(3, df.to_dict(orient='records'), ['X1', 'X2'])
