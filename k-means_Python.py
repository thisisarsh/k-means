import numpy as np
import random

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def kmeans(points, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = points[random.sample(range(len(points)), k)]

    for i in range(max_iterations):
        # Calculate the distances from each point to each centroid
        distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in points])
        
        # Assign each point to the closest centroid
        assignments = np.argmin(distances, axis=1)
        
        # Recalculate the centroids as the mean of the assigned points
        new_centroids = np.array([np.mean(points[assignments == j], axis=0) for j in range(k)])
        
        # Stop if the centroids haven't changed
        if np.array_equal(centroids, new_centroids):
            break
            
        centroids = new_centroids

    return centroids, assignments

# Example usage
points = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
k = 2
centroids, assignments = kmeans(points, k)
print(centroids)
# Output: [[1. 2.]
#          [4. 2.]]
print(assignments)
# Output: [0 0 0 1 1 1]
