#include <cstdio>

#include <cstdlib>

#include <cmath>

#include <ctime>

#include <cuda_runtime.h>

#define BLOCK_SIZE 128

// Data structure to store a point
struct Point {
  float x, y;
};

// Data structure to store the cluster centroids
struct Centroid {
  float x, y;
  int count;
};

// Function to calculate Euclidean distance between two points
__device__ float distance(Point a, Point b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  return sqrt(dx * dx + dy * dy);
}

// Function to update the cluster centroids on the GPU
__global__ void updateCentroids(Point * points, Centroid * centroids, int * assignments, int numPoints, int numCentroids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPoints) return;

  int c = assignments[i];
  atomicAdd( & centroids[c].x, points[i].x);
  atomicAdd( & centroids[c].y, points[i].y);
  atomicAdd( & centroids[c].count, 1);
}

// Function to perform k-means clustering on the GPU
void kmeans(Point * points, Centroid * centroids, int * assignments, int numPoints, int numCentroids, int maxIterations) {
    // Initialize cluster assignments for each point
    for (int i = 0; i < numPoints; i++) {
      int c = rand() % numCentroids;
      assignments[i] = c;
    }

    // Iterate until convergence or maximum iterations reached
    for (int iteration = 0; iteration < maxIterations; iteration++) {
      // Clear the centroids
      for (int i = 0; i < numCentroids; i++) {
        centroids[i].x = 0;
        centroids[i].y = 0;
        centroids[i].count = 0;
      }

      // Update the centroids on the GPU
      int numThreads = BLOCK_SIZE;
      int numBlocks = (numPoints + BLOCK_SIZE - 1) / BLOCK_SIZE;
      updateCentroids << < numBlocks, numThreads >>> (points, centroids, assignments, numPoints, numCentroids);

      // Normalize the centroids
      for (int i = 0; i < numCentroids; i++) {
        if (centroids[i].count > 0) {
          centroids[i].x /= centroids[i].count;
          centroids[i].y /= centroids[i].count;
        }
      }

      // Reassign the points to the closest centroid
      int changed = 0;
      for (int i = 0; i < numPoints; i++) {
        int c = 0;
        float minDistance = distance(points[i], centroids[0]);
        for (int j = 1; j < numCentroids; j++) {
          float d = distance(points[i], centroids[j]);
          if (d < minDistance) {
            c = j;
            minDistance = d;
          }
        }
        if (assignments[i] != c) {
          assignments[i] = c;
          changed++;
        }
      }

      int main() {
        int numPoints = 100000;
        int numCentroids = 10;
        int maxIterations = 100;

        Point * points;
        Centroid * centroids;
        int * assignments;

        // Allocate memory on the host
        cudaMallocHost((void ** ) & points, numPoints * sizeof(Point));
        cudaMallocHost((void ** ) & centroids, numCentroids * sizeof(Centroid));
        cudaMallocHost((void ** ) & assignments, numPoints * sizeof(int));

        // Initialize the data
        srand(time(0));
        for (int i = 0; i < numPoints; i++) {
          points[i].x = rand() % 1000;
          points[i].y = rand() % 1000;
        }
        for (int i = 0; i < numCentroids; i++) {
          centroids[i].x = rand() % 1000;
          centroids[i].y = rand() % 1000;
          centroids[i].count = 0;
        }

        // Perform k-means clustering
        kmeans(points, centroids, assignments, numPoints, numCentroids, maxIterations);

        // Free memory on the host
        cudaFreeHost(points);
        cudaFreeHost(centroids);
        cudaFreeHost(assignments);

        return 0;
      }
