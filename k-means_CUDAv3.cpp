#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_K 1024
#define MAX_D 32

void kmeans_cpu(float *data, float *centroids, int *cluster, int k, int n, int d, int max_iterations);
void print_results(float *data, int *cluster, int k, int n, int d);

int main(int argc, char **argv) {
    int k = 4;    // number of clusters
    int n = 1000; // number of data points
    int d = 2;    // number of dimensions
    
    float *data = (float*)malloc(n * d * sizeof(float));
    float *centroids = (float*)malloc(k * d * sizeof(float));
    int *cluster = (int*)malloc(n * sizeof(int));
    
    // Initialize data and centroids
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            data[i * d + j] = (float)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            centroids[i * d + j] = (float)rand() / RAND_MAX;
        }
    }
    
    // Perform k-means on CPU
    kmeans_cpu(data, centroids, cluster, k, n, d, 100);
    printf("Results from CPU:\n");
    print_results(data, cluster, k, n, d);
    
    // Perform k-means on GPU
    float *d_data, *d_centroids;
    int *d_cluster;
    cudaMalloc((void**)&d_data, n * d * sizeof(float));
    cudaMalloc((void**)&d_centroids, k * d * sizeof(float));
    cudaMalloc((void**)&d_cluster, n * sizeof(int));
    cudaMemcpy(d_data, data, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, k * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster, cluster, n * sizeof(int), cudaMemcpyHostToDevice);
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    for (int i = 0; i < 100; i++) {
        kmeans<<<blocks_per_grid, threads_per_block>>>(d_data, d_centroids, d_cluster, k, n, d);
    }
    cudaMemcpy(centroids, d_centroids, k * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_cluster);
    printf("Results from GPU:\n");
    print_results(data, cluster, k, n, d);
    
    free(data);
    free(centroids);
    free(cluster);
    return 0;
}

/*
This function performs k-means clustering on the CPU. The input parameters are:

data: a pointer to the data points to be clustered
centroids: a pointer to the initial cluster centroids
cluster: a pointer to the array that will contain the assigned cluster for each data point
k: the number of clusters to form
n: the number of data points
d: the number of dimensions of the data points
max_iterations: the maximum number of iterations to perform
The function works by iterating over the following steps until convergence or max_iterations is reached:

Initialize a new array new_centroids to all zeroes and an array counts to all zeroes.
For each data point, assign it to the closest cluster centroid and update the new_centroids and counts arrays accordingly. If the assigned cluster changes, set converged to 0.
If converged is still 1, exit the loop. Otherwise, update the centroids array based on the new_centroids and counts arrays, and repeat the loop.
Once the loop has finished, the cluster array will contain the assigned cluster for each data point, and the centroids array will contain the final cluster centroids.

*/

void kmeans_cpu(float *data, float *centroids, int *cluster, int k, int n, int d, int max_iterations) {
    int *counts = (int*)malloc(k * sizeof(int));
    float *new_centroids = (float*)malloc(k * d * sizeof(float));
    int converged = 0;
    int iter = 0;
    
    while (!converged && iter < max_iterations) {
        converged = 1;
        for (int i = 0; i < k * d; i++) {
            new_centroids[i] = 0.0f;
        }
        for (int i = 0; i < k; i++) {
            counts[i] = 0;
        }
        for (int i = 0; i < n; i++) {
            float min_distance = FLT_MAX;
            int best_cluster = -1;
            for (int j = 0; j < k; j++) {
                float distance = 0.0f;
                for (int l = 0; l < d; l++) {
                    float diff = data[i * d + l] - centroids[j * d + l];
                    distance += diff * diff;
                }
                if (distance < min_distance) {
                    min_distance = distance;
                    best_cluster = j;
                }
            }
            if (cluster[i] != best_cluster) {
                cluster[i] = best_cluster;
                converged = 0;
            }
            for (int j = 0; j < d; j++) {
                new_centroids[best_cluster * d + j] += data[i * d + j];
            }
            counts[best_cluster]++;
        }
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                if (counts[i] > 0) {
                    centroids[i * d + j] = new_centroids[i * d + j] / counts[i];
                }
            }
        }
        iter++;
    }
    free(counts);
    free(new_centroids);
}

__global__ void kmeans(float *data, float *centroids, int *cluster, int k, int n, int d) {
  __shared__ float s_centroids[MAX_K * MAX_D];
  __shared__ int s_counts[MAX_K];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * blockDim.x + tid;

  // Initialize shared memory
  if (tid < k) {
    for (int i = 0; i < d; i++) {
      s_centroids[tid * d + i] = centroids[tid * d + i];
    }
    s_counts[tid] = 0;
  }
  __syncthreads();

  float min_distance = FLT_MAX;
  int best_cluster = -1;

  // Assign data points to nearest centroid
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    for (int j = 0; j < k; j++) {
      float distance = 0.0f;  // corrected line
      for (int l = 0; l < d; l++) {
        float diff = data[i * d + l] - s_centroids[j * d + l];
        distance += diff * diff;
      }
      if (distance < min_distance) {
        min_distance = distance;
        best_cluster = j;
      }
    }
    cluster[i] = best_cluster;
    s_counts[best_cluster]++;
    min_distance = FLT_MAX;
    best_cluster = -1;
  }
  __syncthreads();

  // Update centroids
  for (int i = tid; i < k; i += blockDim.x * gridDim.x) {
    for (int j = 0; j < d; j++) {
      s_centroids[i * d + j] = 0.0f;  // corrected line
    }
    for (int j = 0; j < n; j++) {
      if (cluster[j] == i) {
        for (int l = 0; l < d; l++) {
          s_centroids[i * d + l] += data[j * d + l];
        }
      }
    }
    if (s_counts[i] > 0) {
      for (int j = 0; j < d; j++) {
        s_centroids[i * d + j] /= s_counts[i];
      }
    }
  }
  __syncthreads();

  // Copy centroids back to global memory
  if (tid < k) {
    for (int i = 0; i < d; i++) {
      centroids[tid * d + i] = s_centroids[tid * d + i];
    }
  }
}
