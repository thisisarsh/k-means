#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 1024

__global__ void kmeans_kernel(float *X, float *centers, int *labels,
                              int num_samples, int num_clusters, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;

    float min_dist = FLT_MAX;
    int min_idx = -1;

    for (int i = 0; i < num_clusters; i++) {
        float dist = 0;
        for (int j = 0; j < num_features; j++) {
            float diff = X[idx * num_features + j] - centers[i * num_features + j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }
    labels[idx] = min_idx;
}

void kmeans_on_cuda(float *X, float *centers, int *labels,
                    int num_samples, int num_clusters, int num_features) {
    int num_threads = BLOCK_SIZE;
    int num_blocks = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kmeans_kernel<<<num_blocks, num_threads>>>(X, centers, labels,
                                                num_samples, num_clusters, num_features);
}

int main() {
    // Initialize data and allocate device memory
    int num_samples = 1000;
    int num_clusters = 10;
    int num_features = 100;

    float *X;
    cudaMalloc((void**)&X, num_samples * num_features * sizeof(float));

    float *centers;
    cudaMalloc((void**)&centers, num_clusters * num_features * sizeof(float));

    int *labels;
    cudaMalloc((void**)&labels, num_samples * sizeof(int));

    // Call k-means on CUDA
    kmeans_on_cuda(X, centers, labels, num_samples, num_clusters, num_features);

    // Clean up device memory
    cudaFree(X);
    cudaFree(centers);
    cudaFree(labels);

    return 0;
}
