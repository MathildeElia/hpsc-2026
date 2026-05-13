#include <cstdio>
#include <cstdlib>

__global__ void bucket_count(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
      atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void bucket_sort(int *key, int *bucket, int range) {
  int j = 0;

  for (int i = 0; i < range; i++) {
      for (; bucket[i] > 0; bucket[i]--) {
          key[j++] = i;
      }
  }
}

int main() {
  int n = 50;
  int range = 5;

  int *key;
  int *bucket;

  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i = 0; i < n; i++) {
      key[i] = rand() % range;
      printf("%d ", key[i]);
  }
  printf("\n");

  for (int i = 0; i < range; i++) {
      bucket[i] = 0;
  }

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  bucket_count<<<blocks, threads>>>(key, bucket, n);

  cudaDeviceSynchronize();

  bucket_sort<<<1,1>>>(key, bucket, range);

  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
      printf("%d ", key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}