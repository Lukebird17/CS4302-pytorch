#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cublas_v2.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_M 128
#define BLOCK_SIZE_N 128
#define BLOCK_SIZE_K 8
#define THREAD_SIZE_M 8
#define THREAD_SIZE_N 8

// 检查CUDA错误并自动处理非连续tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define ENSURE_CONTIGUOUS(x) (x.is_contiguous() ? x : x.contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x)

// ==================== 高性能GEMM核心实现 ====================

// 高性能GEMM kernel - 完全按照/hy-tmp/gemm.cu的实现
template<typename T>
__global__ void gemm_kernel_optimized(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int BK_PADDED = BK + 1;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    float res_reg[TM][TN] = {0.0f};
    float4 load_a_reg;
    float4 load_b_reg;
    float frag_a[TM];
    float frag_b[TN];

    __shared__ float As[2][BM][BK_PADDED];
    __shared__ float Bs[2][BK_PADDED][BN];

    int load_a_row = tid / (BK / 4);
    int load_a_col = (tid % (BK / 4)) * 4;
    int load_b_row = tid / (BN / 4);
    int load_b_col = (tid % (BN / 4)) * 4;

    const T* A_ptr = A + by * BM * K;
    const T* B_ptr = B + bx * BN;

    // Prologue
    {
        if (by * BM + load_a_row < M && 0 + load_a_col < K) {
            load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * K + load_a_col)[0];
        } else {
            load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if (0 + load_b_row < K && bx * BN + load_b_col < N) {
            load_b_reg = reinterpret_cast<const float4*>(B_ptr + load_b_row * N + load_b_col)[0];
        } else {
            load_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        As[0][load_a_row][load_a_col + 0] = load_a_reg.x;
        As[0][load_a_row][load_a_col + 1] = load_a_reg.y;
        As[0][load_a_row][load_a_col + 2] = load_a_reg.z;
        As[0][load_a_row][load_a_col + 3] = load_a_reg.w;

        Bs[0][load_b_row][load_b_col + 0] = load_b_reg.x;
        Bs[0][load_b_row][load_b_col + 1] = load_b_reg.y;
        Bs[0][load_b_row][load_b_col + 2] = load_b_reg.z;
        Bs[0][load_b_row][load_b_col + 3] = load_b_reg.w;

        __syncthreads();
    }

    int write_stage_idx = 1;
    int read_stage_idx = 0;

    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        if (next_k < K) {
            if (by * BM + load_a_row < M && next_k + load_a_col < K) {
                load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * K + next_k + load_a_col)[0];
            } else {
                load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            if (next_k + load_b_row < K && bx * BN + load_b_col < N) {
                load_b_reg = reinterpret_cast<const float4*>(B_ptr + (next_k + load_b_row) * N + load_b_col)[0];
            } else {
                load_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                frag_a[r] = As[read_stage_idx][ty * TM + r][i];
            }
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                frag_b[c] = Bs[read_stage_idx][i][tx * TN + c];
            }

            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    res_reg[r][c] += frag_a[r] * frag_b[c];
                }
            }
        }

        __syncthreads();

        if (next_k < K) {
            As[write_stage_idx][load_a_row][load_a_col + 0] = load_a_reg.x;
            As[write_stage_idx][load_a_row][load_a_col + 1] = load_a_reg.y;
            As[write_stage_idx][load_a_row][load_a_col + 2] = load_a_reg.z;
            As[write_stage_idx][load_a_row][load_a_col + 3] = load_a_reg.w;

            Bs[write_stage_idx][load_b_row][load_b_col + 0] = load_b_reg.x;
            Bs[write_stage_idx][load_b_row][load_b_col + 1] = load_b_reg.y;
            Bs[write_stage_idx][load_b_row][load_b_col + 2] = load_b_reg.z;
            Bs[write_stage_idx][load_b_row][load_b_col + 3] = load_b_reg.w;
        }

        __syncthreads();

        read_stage_idx ^= 1;
        write_stage_idx ^= 1;
    }

    // 写回
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        int global_row = by * BM + ty * TM + r;
        
        if (global_row < M) {
            T* C_row_ptr = C + global_row * N;
            
            #pragma unroll
            for (int c_vec = 0; c_vec < TN / 4; ++c_vec) {
                int global_col_start = bx * BN + tx * TN + c_vec * 4;
                
                if (global_col_start + 3 < N) {
                    float4 P_vec;
                    P_vec.x = res_reg[r][c_vec * 4 + 0];
                    P_vec.y = res_reg[r][c_vec * 4 + 1];
                    P_vec.z = res_reg[r][c_vec * 4 + 2];
                    P_vec.w = res_reg[r][c_vec * 4 + 3];
                    
                    float4* C_vec_ptr = reinterpret_cast<float4*>(C_row_ptr + global_col_start);
                    
                    if (beta != 0.0f) {
                        float4 C_old_vec = *C_vec_ptr;
                        C_old_vec.x = alpha * P_vec.x + beta * C_old_vec.x;
                        C_old_vec.y = alpha * P_vec.y + beta * C_old_vec.y;
                        C_old_vec.z = alpha * P_vec.z + beta * C_old_vec.z;
                        C_old_vec.w = alpha * P_vec.w + beta * C_old_vec.w;
                        *C_vec_ptr = C_old_vec;
                    } else {
                        P_vec.x *= alpha; P_vec.y *= alpha;
                        P_vec.z *= alpha; P_vec.w *= alpha;
                        *C_vec_ptr = P_vec;
                    }
                } else {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int global_col = global_col_start + c;
                        if (global_col < N) {
                            int idx = global_row * N + global_col;
                            float val = res_reg[r][c_vec * 4 + c];
                            
                            if (beta != 0.0f) {
                                C[idx] = alpha * val + beta * C[idx];
                            } else {
                                C[idx] = alpha * val;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ==================== 算子融合实现 ====================

// GEMM + Bias
template<typename T>
__global__ void gemm_bias_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    T* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

// GEMM + Bias + GELU融合
template<typename T>
__device__ inline T gelu_activation(T x) {
    const T sqrt_2_over_pi = 0.7978845608028654;
    const T coeff = 0.044715;
    T x_cubed = x * x * x;
    T inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5 * x * (1.0 + tanh(inner));
}

template<typename T>
__global__ void gemm_bias_gelu_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    T* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        T val = sum + bias[col];
        C[row * N + col] = gelu_activation(val);
    }
}

// 简单高效的后处理kernel: Bias + Add + LayerNorm
template<typename T>
__global__ void postprocess_bias_add_layernorm(
    const T* __restrict__ gemm_out,
    const T* __restrict__ bias,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const T* g_ptr = gemm_out + row * N;
    const T* r_ptr = residual + row * N;
    T* o_ptr = output + row * N;
    
    // 1. Bias + Add，同时计算mean
    T sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T val = g_ptr[i] + bias[i] + r_ptr[i];
        o_ptr[i] = val;  // 临时存储
        sum += val;
    }
    
    // Reduce mean
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ T smean;
    if (threadIdx.x == 0) smean = sum / N;
    __syncthreads();
    T mean = smean;
    
    // 2. 计算variance
    sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T diff = o_ptr[i] - mean;
        sum += diff * diff;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ T svar;
    if (threadIdx.x == 0) svar = sum / N;
    __syncthreads();
    
    // 3. LayerNorm
    T inv_std = rsqrtf(svar + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T normalized = (o_ptr[i] - mean) * inv_std;
        o_ptr[i] = gamma[i] * normalized + beta[i];
    }
}

// Bias + GELU + Add + LayerNorm
template<typename T>
__global__ void postprocess_bias_gelu_add_layernorm(
    const T* __restrict__ gemm_out,
    const T* __restrict__ bias,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const T* g_ptr = gemm_out + row * N;
    const T* r_ptr = residual + row * N;
    T* o_ptr = output + row * N;
    
    // 1. Bias + GELU + Add
    T sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T val = gelu_activation(g_ptr[i] + bias[i]) + r_ptr[i];
        o_ptr[i] = val;
        sum += val;
    }
    
    // Reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ T smean;
    if (threadIdx.x == 0) smean = sum / N;
    __syncthreads();
    T mean = smean;
    
    // 2. Variance
    sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T diff = o_ptr[i] - mean;
        sum += diff * diff;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ T svar;
    if (threadIdx.x == 0) svar = sum / N;
    __syncthreads();
    
    // 3. LayerNorm
    T inv_std = rsqrtf(svar + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T normalized = (o_ptr[i] - mean) * inv_std;
        o_ptr[i] = gamma[i] * normalized + beta[i];
    }
}

// GEMM + Bias + Residual Add + LayerNorm融合
// 这是BERT中Attention输出和FFN输出的典型模式
template<typename T>
__global__ void gemm_bias_add_layernorm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    const T* __restrict__ residual,  // 残差连接
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int M, int N, int K,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const T* a_row = A + row * K;
    T* out_row = output + row * N;
    const T* res_row = residual + row * N;
    
    // 1. 计算GEMM + Bias (每个线程处理一部分列)
    extern __shared__ T shared_mem[];
    T* temp = shared_mem;  // 临时存储GEMM结果
    
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += a_row[k] * B[k * N + col];
        }
        temp[col] = sum + bias[col] + res_row[col];  // GEMM + Bias + Residual
    }
    __syncthreads();
    
    // 2. 计算均值
    T sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += temp[i];
    }
    
    // Warp级reduction
    __shared__ T shared_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) shared_sum[wid] = sum;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = shared_sum[threadIdx.x];
    } else {
        sum = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    __shared__ T mean_val;
    if (threadIdx.x == 0) {
        mean_val = sum / N;
    }
    __syncthreads();
    T mean = mean_val;
    
    // 3. 计算方差
    T var = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T diff = temp[i] - mean;
        var += diff * diff;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    
    if (lane == 0) shared_sum[wid] = var;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        var = shared_sum[threadIdx.x];
    } else {
        var = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
    }
    
    __shared__ T std_val;
    if (threadIdx.x == 0) {
        std_val = sqrt(var / N + eps);
    }
    __syncthreads();
    T std = std_val;
    
    // 4. 归一化并应用gamma/beta
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        out_row[i] = gamma[i] * ((temp[i] - mean) / std) + beta[i];
    }
}

// GEMM + Bias + GELU + Residual Add + LayerNorm融合
// 用于FFN: Linear1+GELU -> Linear2+Bias -> Add & LayerNorm
template<typename T>
__global__ void gemm_bias_gelu_add_layernorm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const T* __restrict__ bias,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int M, int N, int K,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const T* a_row = A + row * K;
    T* out_row = output + row * N;
    const T* res_row = residual + row * N;
    
    extern __shared__ T shared_mem[];
    T* temp = shared_mem;
    
    // 1. GEMM + Bias + GELU
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        T sum = 0;
        for (int k = 0; k < K; k++) {
            sum += a_row[k] * B[k * N + col];
        }
        T val = sum + bias[col];
        temp[col] = gelu_activation(val) + res_row[col];  // GELU + Residual
    }
    __syncthreads();
    
    // 2-4. LayerNorm (与上面相同)
    T sum_val = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum_val += temp[i];
    }
    
    __shared__ T shared_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    }
    
    if (lane == 0) shared_sum[wid] = sum_val;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum_val = shared_sum[threadIdx.x];
    } else {
        sum_val = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
    }
    
    __shared__ T mean_val;
    if (threadIdx.x == 0) {
        mean_val = sum_val / N;
    }
    __syncthreads();
    T mean = mean_val;
    
    T var = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T diff = temp[i] - mean;
        var += diff * diff;
    }
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    
    if (lane == 0) shared_sum[wid] = var;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        var = shared_sum[threadIdx.x];
    } else {
        var = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
    }
    
    __shared__ T std_val;
    if (threadIdx.x == 0) {
        std_val = sqrt(var / N + eps);
    }
    __syncthreads();
    T std = std_val;
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        out_row[i] = gamma[i] * ((temp[i] - mean) / std) + beta[i];
    }
}

// LayerNorm融合算子
template<typename T>
__global__ void layernorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    int M, int N,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    
    const T* x = input + row * N;
    T* y = output + row * N;
    
    // 计算均值
    T sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += x[i];
    }
    
    __shared__ T shared_sum[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp级别的reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane == 0) shared_sum[wid] = sum;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = shared_sum[threadIdx.x];
    } else {
        sum = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    __shared__ T mean_val;
    if (threadIdx.x == 0) {
        mean_val = sum / N;
    }
    __syncthreads();
    
    T mean = mean_val;
    
    // 计算方差
    T var = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T diff = x[i] - mean;
        var += diff * diff;
    }
    
    // Warp级别的reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    
    if (lane == 0) shared_sum[wid] = var;
    __syncthreads();
    
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        var = shared_sum[threadIdx.x];
    } else {
        var = 0;
    }
    
    if (wid == 0) {
        for (int offset = (blockDim.x / WARP_SIZE) / 2; offset > 0; offset /= 2) {
            var += __shfl_down_sync(0xffffffff, var, offset);
        }
    }
    
    __shared__ T std_val;
    if (threadIdx.x == 0) {
        std_val = sqrt(var / N + eps);
    }
    __syncthreads();
    
    T std = std_val;
    
    // 归一化并应用缩放和偏移
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        y[i] = gamma[i] * ((x[i] - mean) / std) + beta[i];
    }
}

// ==================== PyTorch接口函数 ====================

torch::Tensor custom_gemm(
    torch::Tensor A,
    torch::Tensor B,
    float alpha,
    float beta
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    // 强制连续化，保证内存对齐
    A = ENSURE_CONTIGUOUS(A);
    B = ENSURE_CONTIGUOUS(B);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 创建输出时确保对齐
    auto options = A.options();
    auto C = torch::empty({M, N}, options);
    
    // 使用优化的tile配置
    const int BM = 128;
    const int BN = 128;
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    if (A.dtype() == torch::kFloat32) {
        gemm_kernel_optimized<float><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K, alpha, beta
        );
    } else {
        throw std::runtime_error("Unsupported data type");
    }
    
    return C;
}

torch::Tensor custom_gemm_bias(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    
    // 确保输入是连续的
    A = ENSURE_CONTIGUOUS(A);
    B = ENSURE_CONTIGUOUS(B);
    bias = ENSURE_CONTIGUOUS(bias);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    if (A.dtype() == torch::kFloat32) {
        gemm_bias_kernel<float><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            bias.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    }
    
    return C;
}

torch::Tensor custom_gemm_bias_gelu(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    
    // 确保输入是连续的
    A = ENSURE_CONTIGUOUS(A);
    B = ENSURE_CONTIGUOUS(B);
    bias = ENSURE_CONTIGUOUS(bias);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    if (A.dtype() == torch::kFloat32) {
        gemm_bias_gelu_kernel<float><<<grid, block>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            bias.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
    }
    
    return C;
}

torch::Tensor custom_layernorm(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    // 确保输入是连续的
    input = ENSURE_CONTIGUOUS(input);
    gamma = ENSURE_CONTIGUOUS(gamma);
    beta = ENSURE_CONTIGUOUS(beta);
    
    int M = input.size(0);
    int N = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    int threads = 256;
    int blocks = M;
    
    if (input.dtype() == torch::kFloat32) {
        layernorm_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            M, N, eps
        );
    }
    
    return output;
}

// GEMM + Bias + Residual + LayerNorm
torch::Tensor custom_gemm_bias_add_layernorm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    A = ENSURE_CONTIGUOUS(A);
    B = ENSURE_CONTIGUOUS(B);
    bias = ENSURE_CONTIGUOUS(bias);
    residual = ENSURE_CONTIGUOUS(residual);
    gamma = ENSURE_CONTIGUOUS(gamma);
    beta = ENSURE_CONTIGUOUS(beta);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 步骤1: 优化GEMM
    auto gemm_out = torch::empty({M, N}, A.options());
    
    const int BM = 128;
    const int BN = 128;
    dim3 block1(16, 16);
    dim3 grid1((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_kernel_optimized<float><<<grid1, block1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        gemm_out.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f
    );
    
    // 步骤2: 后处理 (Bias + Add + LayerNorm)
    auto output = torch::empty({M, N}, A.options());
    
    dim3 block2(256);
    dim3 grid2(M);
    
    postprocess_bias_add_layernorm<float><<<grid2, block2>>>(
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, eps
    );
    
    return output;
}

// GEMM + Bias + GELU + Residual + LayerNorm
torch::Tensor custom_gemm_bias_gelu_add_layernorm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(bias);
    CHECK_INPUT(residual);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    
    A = ENSURE_CONTIGUOUS(A);
    B = ENSURE_CONTIGUOUS(B);
    bias = ENSURE_CONTIGUOUS(bias);
    residual = ENSURE_CONTIGUOUS(residual);
    gamma = ENSURE_CONTIGUOUS(gamma);
    beta = ENSURE_CONTIGUOUS(beta);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 步骤1: 优化GEMM
    auto gemm_out = torch::empty({M, N}, A.options());
    
    const int BM = 128;
    const int BN = 128;
    dim3 block1(16, 16);
    dim3 grid1((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_kernel_optimized<float><<<grid1, block1>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        gemm_out.data_ptr<float>(),
        M, N, K, 1.0f, 0.0f
    );
    
    // 步骤2: 后处理 (Bias + GELU + Add + LayerNorm)
    auto output = torch::empty({M, N}, A.options());
    
    dim3 block2(256);
    dim3 grid2(M);
    
    postprocess_bias_gelu_add_layernorm<float><<<grid2, block2>>>(
        gemm_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        residual.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, eps
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &custom_gemm, "Custom GEMM");
    m.def("gemm_bias", &custom_gemm_bias, "Custom GEMM with Bias");
    m.def("gemm_bias_gelu", &custom_gemm_bias_gelu, "Custom GEMM with Bias and GELU");
    m.def("layernorm", &custom_layernorm, "Custom LayerNorm");
    m.def("gemm_bias_add_layernorm", &custom_gemm_bias_add_layernorm, 
          "Custom GEMM + Bias + Residual Add + LayerNorm");
    m.def("gemm_bias_gelu_add_layernorm", &custom_gemm_bias_gelu_add_layernorm,
          "Custom GEMM + Bias + GELU + Residual Add + LayerNorm");
}

