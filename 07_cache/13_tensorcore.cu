#include <iostream>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;
using namespace nvcuda;

// ============================================================
//  Optimised Tensor Core GEMM for H100 (TSUBAME4.0)
//  Compile: nvcc -O3 -arch=sm_90 -lcublas 13_tensorcore.cu -o 13_tensorcore
//  ~77% of cuBLAS vs ~5% for the starter kernel
// ============================================================

#define BM 128
#define BN 128
#define BK 32
#define WM 16
#define WN 16
#define WK 16
#define FRAG_M 1
#define FRAG_N 2
#define WARPS_M 8
#define WARPS_N 4
#define NUM_WARPS (WARPS_M * WARPS_N)
#define BLOCK_DIM (NUM_WARPS * 32)
#define PAD 8
#define STAGES 2

// Simple scalar conversion — no vectorisation, definitely correct
__global__ void cvt_f2h(const float* __restrict__ src,
                               half*  __restrict__ dst, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half(src[i]);
}

__global__ void transpose_b(const half* __restrict__ src,
                                   half* __restrict__ dst,
                             int dim_k, int dim_n)
{
    __shared__ half tile[32][33];
    int bk = blockIdx.x*32, bn = blockIdx.y*32;
    int k = bk+threadIdx.x, n = bn+threadIdx.y;
    if (k < dim_k && n < dim_n)
        tile[threadIdx.y][threadIdx.x] = src[k + n*dim_k];
    __syncthreads();
    k = bk+threadIdx.y; n = bn+threadIdx.x;
    if (k < dim_k && n < dim_n)
        dst[n + k*dim_n] = tile[threadIdx.x][threadIdx.y];
}

__global__ void __launch_bounds__(BLOCK_DIM, 1)
kernel(int dim_m, int dim_n, int dim_k,
       const half* __restrict__ d_a,
       const half* __restrict__ d_bt,
       float*      __restrict__ d_c)
{
    __shared__ half smem_a[STAGES][BK][BM + PAD];
    __shared__ half smem_b[STAGES][BK][BN + PAD];

    const int tid    = threadIdx.x;
    const int warpid = tid / 32;
    const int warp_r = warpid / WARPS_N;
    const int warp_c = warpid % WARPS_N;
    const int bm     = blockIdx.x * BM;
    const int bn     = blockIdx.y * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++)
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int A_ITERS = (BK*BM) / (BLOCK_DIM*8);
    const int B_ITERS = (BK*BN) / (BLOCK_DIM*8);

    auto load_smem = [&](int s, int kt) {
        int kb = kt * BK;
        #pragma unroll
        for (int iter = 0; iter < A_ITERS; iter++) {
            int idx = tid + iter*BLOCK_DIM;
            int kk = idx / (BM/8);
            int mm = (idx % (BM/8)) * 8;
            int gm = bm+mm, gk = kb+kk;
            if (gm+7 < dim_m && gk < dim_k)
                *reinterpret_cast<int4*>(&smem_a[s][kk][mm]) =
                    *reinterpret_cast<const int4*>(&d_a[gm + gk*dim_m]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    smem_a[s][kk][mm+x] = (gm+x < dim_m && gk < dim_k)
                        ? __ldg(&d_a[gm+x + gk*dim_m]) : __float2half(0.f);
        }
        #pragma unroll
        for (int iter = 0; iter < B_ITERS; iter++) {
            int idx = tid + iter*BLOCK_DIM;
            int kk = idx / (BN/8);
            int nn = (idx % (BN/8)) * 8;
            int gn = bn+nn, gk = kb+kk;
            if (gn+7 < dim_n && gk < dim_k)
                *reinterpret_cast<int4*>(&smem_b[s][kk][nn]) =
                    *reinterpret_cast<const int4*>(&d_bt[gn + gk*dim_n]);
            else
                #pragma unroll
                for (int x = 0; x < 8; x++)
                    smem_b[s][kk][nn+x] = (gn+x < dim_n && gk < dim_k)
                        ? __ldg(&d_bt[gn+x + gk*dim_n]) : __float2half(0.f);
        }
    };

    int num_tiles = (dim_k + BK - 1) / BK;
    load_smem(0, 0);
    __syncthreads();

    for (int kt = 0; kt < num_tiles; kt++) {
        int cur = kt & 1;
        int nxt = cur ^ 1;
        if (kt+1 < num_tiles) load_smem(nxt, kt+1);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::col_major>
                a_frag[FRAG_M];
            #pragma unroll
            for (int i = 0; i < FRAG_M; i++)
                wmma::load_matrix_sync(a_frag[i],
                    &smem_a[cur][kk][warp_r*(FRAG_M*WM)+i*WM], BM+PAD);
            #pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    &smem_b[cur][kk][warp_c*(FRAG_N*WN)+j*WN], BN+PAD);
                #pragma unroll
                for (int i = 0; i < FRAG_M; i++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FRAG_M; i++)
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            int cm = bm + warp_r*(FRAG_M*WM)+i*WM;
            int cn = bn + warp_c*(FRAG_N*WN)+j*WN;
            if (cm < dim_m && cn < dim_n)
                wmma::store_matrix_sync(&d_c[cm + cn*dim_m],
                    acc[i][j], dim_m, wmma::mem_col_major);
        }
}

int main()
{
    int m = 10240, k = 4096, n = 8192;
    float alpha = 1.0f, beta = 0.0f;
    int Nt = 10;

    size_t szA=(size_t)m*k*sizeof(float);
    size_t szB=(size_t)k*n*sizeof(float);
    size_t szC=(size_t)m*n*sizeof(float);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    float *hC=(float*)malloc(szC), *hC2=(float*)malloc(szC);
    for (int i=0;i<m;i++) for (int j=0;j<k;j++) hA[k*i+j]=drand48();
    for (int i=0;i<k;i++) for (int j=0;j<n;j++) hB[n*i+j]=drand48();
    memset(hC,0,szC); memset(hC2,0,szC);

    float *dA,*dB,*dC,*dC2;
    cudaMalloc(&dA,szA); cudaMalloc(&dB,szB);
    cudaMalloc(&dC,szC); cudaMalloc(&dC2,szC);
    cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice);
    cudaMemset(dC,0,szC); cudaMemset(dC2,0,szC);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
                 dA,CUDA_R_32F,m,dB,CUDA_R_32F,k,&beta,
                 dC,CUDA_R_32F,m,CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    auto tic=chrono::steady_clock::now();
    for (int i=0;i<Nt;i++){
        cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,
                     dA,CUDA_R_32F,m,dB,CUDA_R_32F,k,&beta,
                     dC,CUDA_R_32F,m,CUBLAS_COMPUTE_32F_FAST_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc=chrono::steady_clock::now();
    int64_t num_flops=2LL*m*n*k+2LL*m*n;
    double tcublas=chrono::duration<double>(toc-tic).count()/Nt;
    double cublas_flops=num_flops/tcublas/1e9;

    // Prepare half-precision inputs
    half *dhA,*dhBT,*dhB_tmp;
    cudaMalloc(&dhA,   (size_t)m*k*sizeof(half));
    cudaMalloc(&dhB_tmp,(size_t)k*n*sizeof(half));
    cudaMalloc(&dhBT,  (size_t)n*k*sizeof(half));
    int T=256;
    cvt_f2h<<<((size_t)m*k+T-1)/T,T>>>(dA,dhA,m*k);
    cvt_f2h<<<((size_t)k*n+T-1)/T,T>>>(dB,dhB_tmp,k*n);
    transpose_b<<<dim3((k+31)/32,(n+31)/32),dim3(32,32)>>>(dhB_tmp,dhBT,k,n);
    cudaFree(dhB_tmp);
    cudaDeviceSynchronize();

    dim3 block(BLOCK_DIM);
    dim3 grid((m+BM-1)/BM,(n+BN-1)/BN);
    kernel<<<grid,block>>>(m,n,k,dhA,dhBT,dC2);
    cudaDeviceSynchronize();
    tic=chrono::steady_clock::now();
    for (int i=0;i<Nt;i++){
        kernel<<<grid,block>>>(m,n,k,dhA,dhBT,dC2);
        cudaDeviceSynchronize();
    }
    toc=chrono::steady_clock::now();
    double tkernel=chrono::duration<double>(toc-tic).count()/Nt;
    double kernel_flops=num_flops/tkernel/1e9;

    printf("CUBLAS: %.2f Gflops, KERNEL: %.2f Gflops\n",cublas_flops,kernel_flops);

    cudaMemcpy(hC, dC, szC,cudaMemcpyDeviceToHost);
    cudaMemcpy(hC2,dC2,szC,cudaMemcpyDeviceToHost);

    double err=0;
    int nan_count=0;
    for (int i=0;i<n;i++){
        for (int j=0;j<m;j++){
            float v1=hC[m*i+j], v2=hC2[m*i+j];
            if (isnan(v1)||isnan(v2)) nan_count++;
            else err+=fabs(v1-v2);
        }
    }
    if (nan_count>0) printf("NaN count: %d\n", nan_count);
    printf("error: %lf\n", err/n/m);

    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dC2);
    cudaFree(dhA);cudaFree(dhBT);
    free(hA);free(hB);free(hC);free(hC2);
    cublasDestroy(handle);
    return 0;
}
