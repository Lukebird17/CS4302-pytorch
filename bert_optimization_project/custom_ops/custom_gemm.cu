/*
 * 自定义GEMM实现 - 针对BERT优化
 * 
 * 目标: 超越或接近cuBLAS的性能
 * 优化技术:
 * 1. Shared Memory Tiling - 减少全局内存访问
 * 2. Register Tiling - 提高计算吞吐量
 * 3. Memory Coalescing - 合并内存访问
 * 4. Bank Conflict避免
 * 5. 针对BERT常见矩阵大小特化 (768, 3072)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ============================================================================
// GEMM Kernel - 基础版本 (C = A * B + bias)
// ============================================================================

/*
 * 矩阵乘法: C[M,N] = A[M,K] * B[K,N]
 * 
 * 优化策略:
 * - Tile大小: 32x32 (适配warp)
 * - 每个线程计算8x8的结果块
 * - 使用shared memory缓存
 */

template<int BLOCK_SIZE = 32>
__global__ void gemm_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load tile into shared memory
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================================
// GEMM Kernel - 优化版本 (使用Register Tiling)
// ============================================================================

/*
 * 高性能GEMM kernel
 * 每个线程计算4x4的输出块，使用寄存器存储
 */
template<int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void gemm_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    // Thread indices
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    
    // Block indices
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Register storage for results
    float accum[TM][TN] = {0.0f};
    
    // Thread's output position
    int threadRowStart = (tid / (BN / TN)) * TM;
    int threadColStart = (tid % (BN / TN)) * TN;
    
    // Main loop over K dimension
    for (int k = 0; k < K; k += BK) {
        // Load A tile into shared memory (coalesced)
        for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
            int row = i / BK;
            int col = i % BK;
            int globalRow = blockRow * BM + row;
            int globalCol = k + col;
            As[row][col] = (globalRow < M && globalCol < K) ? 
                           A[globalRow * K + globalCol] : 0.0f;
        }
        
        // Load B tile into shared memory (coalesced)
        for (int i = tid; i < BK * BN; i += blockDim.x * blockDim.y) {
            int row = i / BN;
            int col = i % BN;
            int globalRow = k + row;
            int globalCol = blockCol * BN + col;
            Bs[row][col] = (globalRow < K && globalCol < N) ? 
                           B[globalRow * N + globalCol] : 0.0f;
        }
        
        __syncthreads();
        
        // Compute using register tiling
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    accum[i][j] += As[threadRowStart + i][kk] * 
                                   Bs[kk][threadColStart + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalRow = blockRow * BM + threadRowStart + i;
            int globalCol = blockCol * BN + threadColStart + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}


// ============================================================================
// GEMM Kernel - 针对BERT 768维度特化
// ============================================================================

/*
 * 专门针对BERT的768维度优化
 * 使用16x16 tile，循环24次遍历K=768
 */

__global__ void gemm_768_kernel(
    const float* __restrict__ A,  // [M, 768]
    const float* __restrict__ B,  // [768, N]
    float* __restrict__ C,        // [M, N]
    int M, int N
) {
    constexpr int K = 768;
    constexpr int TILE_K = 16;  // 改为16，匹配block大小
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    
    // Shared memory: 16x16 + 16x16 = 512 floats = 2KB
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    
    float sum = 0.0f;
    
    // 循环遍历K维度：768 / 16 = 48次
    for (int t = 0; t < K; t += TILE_K) {
        // Load A tile - 每个线程加载1个元素
        if (row < M && (t + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile - 每个线程加载1个元素
        if ((t + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================================
// 向量化GEMM (使用float4)
// ============================================================================

__global__ void gemm_kernel_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 8;
    
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x * 4;  // 每次处理4个
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    for (int k = 0; k < K; k += TILE_K) {
        // Load with vectorization
        if (row < M && k + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        }
        
        if (col < N && k + threadIdx.y < K) {
            float4* b_ptr = (float4*)&B[(k + threadIdx.y) * N + col];
            float4 b_val = *b_ptr;
            Bs[threadIdx.y][threadIdx.x * 4] = b_val.x;
            Bs[threadIdx.y][threadIdx.x * 4 + 1] = b_val.y;
            Bs[threadIdx.y][threadIdx.x * 4 + 2] = b_val.z;
            Bs[threadIdx.y][threadIdx.x * 4 + 3] = b_val.w;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float a_val = As[threadIdx.y][kk];
            sum.x += a_val * Bs[kk][threadIdx.x * 4];
            sum.y += a_val * Bs[kk][threadIdx.x * 4 + 1];
            sum.z += a_val * Bs[kk][threadIdx.x * 4 + 2];
            sum.w += a_val * Bs[kk][threadIdx.x * 4 + 3];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col + 3 < N) {
        float4* c_ptr = (float4*)&C[row * N + col];
        *c_ptr = sum;
    } else if (row < M) {
        if (col < N) C[row * N + col] = sum.x;
        if (col + 1 < N) C[row * N + col + 1] = sum.y;
        if (col + 2 < N) C[row * N + col + 2] = sum.z;
        if (col + 3 < N) C[row * N + col + 3] = sum.w;
    }
}


// ============================================================================
// Python接口包装
// ============================================================================

torch::Tensor custom_gemm(
    torch::Tensor A,
    torch::Tensor B
) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // 选择最优kernel
    if (K == 768) {
        // BERT特化版本：16x16 block
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_768_kernel<<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N
        );
    } else if (N % 4 == 0 && N >= 128) {
        // 向量化版本
        dim3 block(32, 8);
        dim3 grid((N / 4 + 31) / 32, (M + 7) / 8);
        gemm_kernel_vectorized<<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    } else {
        // 通用优化版本
        dim3 block(16, 16);
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        gemm_kernel_optimized<128, 128, 8, 8, 8><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    }
    
    return C;
}


// ============================================================================
// GEMM + Bias + GELU融合
// ============================================================================

__device__ __forceinline__ float fast_gelu(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void gemm_bias_gelu_kernel_768(
    const float* __restrict__ A,  // [M, 768]
    const float* __restrict__ B,  // [768, N]
    const float* __restrict__ bias,  // [N]
    float* __restrict__ C,        // [M, N]
    int M, int N
) {
    constexpr int K = 768;
    constexpr int TILE_K = 16;
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    
    // Shared memory: 16x16 + 16x16 = 512 floats = 2KB
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;
    
    float sum = 0.0f;
    
    // 循环遍历K维度
    for (int t = 0; t < K; t += TILE_K) {
        // Load tiles
        if (row < M && (t + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Fuse: Add bias + GELU
    if (row < M && col < N) {
        sum += bias[col];
        sum = fast_gelu(sum);
        C[row * N + col] = sum;
    }
}

torch::Tensor custom_gemm_bias_gelu(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    if (K == 768) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_bias_gelu_kernel_768<<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            bias.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N
        );
    }
    
    return C;
}


// ============================================================================
// 模块绑定
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_gemm", &custom_gemm, "Custom GEMM implementation");
    m.def("custom_gemm_bias_gelu", &custom_gemm_bias_gelu, 
          "Custom GEMM + Bias + GELU fusion");
}

