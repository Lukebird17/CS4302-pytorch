/*
 * BERT专用高性能GEMM - 基于双缓冲流水线优化
 * 严格按照 /hy-tmp/gemm.cu 的实现思路
 * 
 * 参考优化技术:
 * 1. 双缓冲流水线 (Double Buffering) - 隐藏内存延迟
 * 2. float4向量化加载/存储 - 4x内存带宽
 * 3. Bank Conflict规避 (BK+1 padding)
 * 4. 寄存器级Tiling (TM=8, TN=8)
 * 5. FMA指令优化
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------
// 关键常量定义 (基于标准 GEMM Tiling 策略)
// -----------------------------------------------------------------
// Thread Block 结果矩阵的维度
const int BM = 128; // Block M
const int BN = 128; // Block N
// K 维度步长（8是Bank Conflict高危值，但在此代码中被填充规避）
const int BK = 8;  
// 每个线程累积的 M 维度
const int TM = 8;  
// 每个线程累积的 N 维度
const int TN = 8;  
// Thread Block 线程数：16x16 = 256
// -----------------------------------------------------------------

template <typename scalar_t>
__global__ void my_gemm_double_buffer_kernel(
    int M, int N, int K,
    scalar_t alpha,
    const scalar_t* __restrict__ A, int lda,
    const scalar_t* __restrict__ B, int ldb,
    scalar_t beta,
    scalar_t* __restrict__ C, int ldc) {

    // --- 优化 1: Bank Conflict 规避 ---
    // 将 BK 维度从 8 填充到 9，改变访问步长，消除对齐冲突
    const int BK_PADDED = BK + 1; 

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    // --- 寄存器 ---
    // 假设 scalar_t 为 float (FP32)
    float res_reg[TM][TN] = {0.0f}; 
    float4 load_a_reg;
    float4 load_b_reg;
    float frag_a[TM];
    float frag_b[TN];

    // --- Shared Memory (使用 BK_PADDED 规避 Bank Conflict) ---
    __shared__ float As[2][BM][BK_PADDED]; 
    __shared__ float Bs[2][BK_PADDED][BN];

    // --- 加载索引计算 (将 256 线程映射到 M*K 和 K*N 矩阵的加载) ---
    // BM*BK (128*8) / 4 = 256
    int load_a_row = tid / (BK / 4);
    int load_a_col = (tid % (BK / 4)) * 4;

    // BK*BN (8*128) / 4 = 256
    int load_b_row = tid / (BN / 4);
    int load_b_col = (tid % (BN / 4)) * 4;

    // 移动指针到当前Block的起点
    const scalar_t* A_ptr = A + by * BM * lda;
    const scalar_t* B_ptr = B + bx * BN;

    // --- Prologue: 加载第0个Tile (stage 0) ---
    {
        // 加载 A (M x K)
        if (by * BM + load_a_row < M && 0 + load_a_col < K) {
            load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * lda + load_a_col)[0];
        } else {
            load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 加载 B (K x N)
        if (0 + load_b_row < K && bx * BN + load_b_col < N) {
            load_b_reg = reinterpret_cast<const float4*>(B_ptr + load_b_row * ldb + load_b_col)[0];
        } else {
            load_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 写入 Shared Memory (使用 BK_PADDED 维度作为 stride)
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

    // --- Main Loop: 沿 K 维度累积 ---
    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        // 1. Prefetch Next Tile (在当前计算周期内异步加载下一块)
        if (next_k < K) {
            // 加载 A (边界检查和零填充)
            if (by * BM + load_a_row < M && next_k + load_a_col < K) {
                load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * lda + next_k + load_a_col)[0];
            } else {
                load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            // 加载 B (边界检查和零填充)
            if (next_k + load_b_row < K && bx * BN + load_b_col < N) {
                load_b_reg = reinterpret_cast<const float4*>(B_ptr + (next_k + load_b_row) * ldb + load_b_col)[0];
            } else {
                load_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // 2. Compute Current Tile (使用 read_stage_idx 上的数据)
        #pragma unroll
        for (int i = 0; i < BK; ++i) { // 沿 K 维度进行 8 次乘加
            
            // 从 Shared Memory 加载到 Register fragments
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                frag_a[r] = As[read_stage_idx][ty * TM + r][i];
            }
            #pragma unroll
            for (int c = 0; c < TN; ++c) {
                frag_b[c] = Bs[read_stage_idx][i][tx * TN + c];
            }
            
            // FMA (Fused Multiply-Add)
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    res_reg[r][c] += frag_a[r] * frag_b[c];
                }
            }
        }

        // 3. Store Prefetched Tile to Shared (使用 write_stage_idx 上的缓冲区)
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

        // 交换读写缓冲区
        read_stage_idx ^= 1;
        write_stage_idx ^= 1;
    }

    // --- 优化 2: 向量化写回 (Vectorized Write Back) ---
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        int global_row = by * BM + ty * TM + r;
        
        if (global_row < M) {
            // C_row_ptr 指向当前行起始地址
            scalar_t* C_row_ptr = C + global_row * ldc;
            
            // TN=8，分成 c_vec = 0, 1 两个 float4 写入
            #pragma unroll
            for (int c_vec = 0; c_vec < TN / 4; ++c_vec) { 
                int global_col_start = bx * BN + tx * TN + c_vec * 4;
                
                // 1. 优先尝试向量化写回 (要求一次写入 4 个元素)
                if (global_col_start + 3 < N) {
                    
                    // 准备 (A@B) 结果向量 P_vec
                    float4 P_vec;
                    P_vec.x = res_reg[r][c_vec * 4 + 0];
                    P_vec.y = res_reg[r][c_vec * 4 + 1];
                    P_vec.z = res_reg[r][c_vec * 4 + 2];
                    P_vec.w = res_reg[r][c_vec * 4 + 3];
                    
                    float4* C_vec_ptr = reinterpret_cast<float4*>(C_row_ptr + global_col_start);
                    
                    if (beta != 0.0f) {
                        // 读-改-写 (RMW) 模式：向量化读取 C_old
                        float4 C_old_vec = *C_vec_ptr;
                        
                        // 应用乘加 (FMA)
                        C_old_vec.x = alpha * P_vec.x + beta * C_old_vec.x;
                        C_old_vec.y = alpha * P_vec.y + beta * C_old_vec.y;
                        C_old_vec.z = alpha * P_vec.z + beta * C_old_vec.z;
                        C_old_vec.w = alpha * P_vec.w + beta * C_old_vec.w;
                        
                        // 向量化写回
                        *C_vec_ptr = C_old_vec;
                    } else {
                        // 纯 MM 模式：只写 (无需读取 C)
                        P_vec.x *= alpha; P_vec.y *= alpha;
                        P_vec.z *= alpha; P_vec.w *= alpha;
                        *C_vec_ptr = P_vec;
                    }

                } else {
                    // 2. 边界条件或无法对齐，退回到标量写回
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int global_col = global_col_start + c;
                        if (global_col < N) {
                            int idx = global_row * ldc + global_col;
                            float val = res_reg[r][c_vec * 4 + c];
                            
                            if (beta != 0.0f) {
                                C[idx] = alpha * val + beta * C[idx];
                            } else {
                                C[idx] = alpha * val;
                            }
                        }
                    }
                }
            } // end c_vec loop (TN/4)
        } // end global_row < M check
    } // end r loop (TM)
}


// ============================================================================
// 融合GEMM + Bias + GELU Kernel
// ============================================================================

__device__ __forceinline__ float fast_gelu(float x) {
    float x_cubed = x * x * x;
    float inner = __fmaf_rn(0.044715f, x_cubed, x);
    inner = __fmul_rn(0.7978845608f, inner);
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, tanhf(inner)));
}

template <typename scalar_t>
__global__ void gemm_bias_gelu_kernel(
    int M, int N, int K,
    const scalar_t* __restrict__ A, int lda,
    const scalar_t* __restrict__ B, int ldb,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ C, int ldc) {

    const int BK_PADDED = BK + 1;
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

    const scalar_t* A_ptr = A + by * BM * lda;
    const scalar_t* B_ptr = B + bx * BN;

    // Prologue
    {
        if (by * BM + load_a_row < M && load_a_col < K) {
            load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * lda + load_a_col)[0];
        } else {
            load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        if (load_b_row < K && bx * BN + load_b_col < N) {
            load_b_reg = reinterpret_cast<const float4*>(B_ptr + load_b_row * ldb + load_b_col)[0];
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

    // Main Loop
    for (int k = 0; k < K; k += BK) {
        int next_k = k + BK;

        if (next_k < K) {
            if (by * BM + load_a_row < M && next_k + load_a_col < K) {
                load_a_reg = reinterpret_cast<const float4*>(A_ptr + load_a_row * lda + next_k + load_a_col)[0];
            } else {
                load_a_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            if (next_k + load_b_row < K && bx * BN + load_b_col < N) {
                load_b_reg = reinterpret_cast<const float4*>(B_ptr + (next_k + load_b_row) * ldb + load_b_col)[0];
            } else {
                load_b_reg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // Compute
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
                    res_reg[r][c] = __fmaf_rn(frag_a[r], frag_b[c], res_reg[r][c]);
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

    // 融合Bias+GELU写回
    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        int global_row = by * BM + ty * TM + r;
        if (global_row < M) {
            scalar_t* C_row = C + global_row * ldc;
            #pragma unroll
            for (int c_vec = 0; c_vec < TN / 4; ++c_vec) {
                int global_col = bx * BN + tx * TN + c_vec * 4;
                if (global_col + 3 < N) {
                    float4 result_vec;
                    result_vec.x = fast_gelu(res_reg[r][c_vec * 4 + 0] + bias[global_col + 0]);
                    result_vec.y = fast_gelu(res_reg[r][c_vec * 4 + 1] + bias[global_col + 1]);
                    result_vec.z = fast_gelu(res_reg[r][c_vec * 4 + 2] + bias[global_col + 2]);
                    result_vec.w = fast_gelu(res_reg[r][c_vec * 4 + 3] + bias[global_col + 3]);
                    reinterpret_cast<float4*>(C_row + global_col)[0] = result_vec;
                } else {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        int col = global_col + c;
                        if (col < N) {
                            C_row[col] = fast_gelu(res_reg[r][c_vec * 4 + c] + bias[col]);
                        }
                    }
                }
            }
        }
    }
}


// ============================================================================
// Python接口
// ============================================================================

torch::Tensor custom_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Matrices must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only FP32 supported");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // 使用 alpha=1.0, beta=0.0 作为默认值
    my_gemm_double_buffer_kernel<float><<<grid, block>>>(
        M, N, K,
        1.0f,
        A.data_ptr<float>(), K,
        B.data_ptr<float>(), N,
        0.0f,
        C.data_ptr<float>(), N
    );
    
    return C;
}

torch::Tensor custom_gemm_bias_gelu(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Matrices must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias.is_cuda(), "Must be CUDA tensors");
    TORCH_CHECK(bias.size(0) == B.size(1), "Bias dimension mismatch");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Tensors must be contiguous");
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    gemm_bias_gelu_kernel<float><<<grid, block>>>(
        M, N, K,
        A.data_ptr<float>(), K,
        B.data_ptr<float>(), N,
        bias.data_ptr<float>(),
        C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_gemm", &custom_gemm, "Double-buffered high-performance GEMM");
    m.def("custom_gemm_bias_gelu", &custom_gemm_bias_gelu, 
          "Double-buffered GEMM+Bias+GELU fusion");
}
