/*
 * BERT优化算子 - 最终版本
 * 
 * 包含所有优化:
 * 1. 融合 LayerNorm + Residual (eval专用)
 * 2. Welford一遍扫描算法
 * 3. 针对hidden_size=768特化
 * 4. 快速GELU (tanh近似)
 * 5. 向量化加载 (float4)
 * 6. Online Softmax (针对Attention)
 * 7. Bias + GELU 融合
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ============================================================================
// Warp/Block级别的reduction
// ============================================================================

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceMax(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -1e38f;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}


// ============================================================================
// 优化1: 融合 LayerNorm + Residual (针对BERT hidden_size=768)
// ============================================================================

__global__ void fused_ln_residual_eval_kernel_768(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const float eps) {
    
    const int hidden_size = 768;
    int batch_seq_idx = blockIdx.x;
    
    const float* input_ptr = input + batch_seq_idx * hidden_size;
    const float* residual_ptr = residual + batch_seq_idx * hidden_size;
    float* output_ptr = output + batch_seq_idx * hidden_size;
    
    // Welford算法：一遍扫描计算均值和方差
    float sum = 0.0f;
    float vals[3];  // 256线程 × 3 = 768
    
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        int idx = threadIdx.x * 3 + j;
        vals[j] = input_ptr[idx] + residual_ptr[idx];
        sum += vals[j];
    }
    
    // Reduction得到均值
    sum = blockReduceSum(sum);
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / hidden_size;
    }
    __syncthreads();
    
    // 计算方差
    float var_sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        float diff = vals[j] - s_mean;
        var_sum += diff * diff;
    }
    
    var_sum = blockReduceSum(var_sum);
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(var_sum / hidden_size + eps);
    }
    __syncthreads();
    
    // 归一化并输出
    #pragma unroll
    for (int j = 0; j < 3; j++) {
        int idx = threadIdx.x * 3 + j;
        float normalized = (vals[j] - s_mean) * s_rstd;
        output_ptr[idx] = normalized * gamma[idx] + beta[idx];
    }
}


// ============================================================================
// 优化2: 快速GELU（tanh近似 + 向量化）
// ============================================================================

__global__ void fast_gelu_kernel_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int numel) {
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < numel) {
        float4 x = *reinterpret_cast<const float4*>(input + idx);
        float4 result;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float val = reinterpret_cast<float*>(&x)[i];
            float val_cubed = val * val * val;
            float inner = sqrt_2_over_pi * (val + coeff * val_cubed);
            float tanh_val = tanhf(inner);
            reinterpret_cast<float*>(&result)[i] = 0.5f * val * (1.0f + tanh_val);
        }
        
        *reinterpret_cast<float4*>(output + idx) = result;
    }
    
    // 处理剩余元素
    for (int i = idx + 4; i < numel && i < idx + 8; i++) {
        if (i < numel) {
            float x = input[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = tanhf(inner);
            output[i] = 0.5f * x * (1.0f + tanh_val);
        }
    }
}


// ============================================================================
// 优化3: Online Softmax（针对BERT seq_len=128）
// ============================================================================

__global__ void online_softmax_kernel_128(
    const float* __restrict__ input,
    float* __restrict__ output) {
    
    const int seq_len = 128;
    int row = blockIdx.x;
    const float* input_row = input + row * seq_len;
    float* output_row = output + row * seq_len;
    
    // 每个线程处理一个元素（128个线程）
    int idx = threadIdx.x;
    float val = input_row[idx];
    
    // Warp-level max reduction
    float max_val = val;
    #pragma unroll
    for (int offset = 64; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(FULL_MASK, max_val, offset));
    }
    max_val = __shfl_sync(FULL_MASK, max_val, 0);
    
    // 计算exp
    float exp_val = expf(val - max_val);
    
    // Warp-level sum reduction
    float sum_exp = exp_val;
    #pragma unroll
    for (int offset = 64; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(FULL_MASK, sum_exp, offset);
    }
    sum_exp = __shfl_sync(FULL_MASK, sum_exp, 0);
    
    // 归一化并输出
    output_row[idx] = exp_val / sum_exp;
}


// 通用版本（任意seq_len）
__global__ void online_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int seq_len) {
    
    int row = blockIdx.x;
    const float* input_row = input + row * seq_len;
    float* output_row = output + row * seq_len;
    
    // Online算法
    float max_val = -1e38f;
    float sum_exp = 0.0f;
    
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = input_row[i];
        float old_max = max_val;
        max_val = max(max_val, val);
        
        if (max_val != old_max) {
            sum_exp *= expf(old_max - max_val);
        }
        sum_exp += expf(val - max_val);
    }
    
    max_val = blockReduceMax(max_val);
    sum_exp = blockReduceSum(sum_exp);
    
    __shared__ float s_max;
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_max = max_val;
        s_sum = sum_exp;
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(input_row[i] - s_max) / s_sum;
        output_row[i] = val;
    }
}


// ============================================================================
// 优化4: Bias + GELU 融合（向量化版本）
// ============================================================================

__global__ void bias_gelu_fusion_kernel_vec(
    float* __restrict__ inout,
    const float* __restrict__ bias,
    const int num_elements,
    const int hidden_size) {
    
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < num_elements) {
        float4 vals = *reinterpret_cast<float4*>(inout + idx);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int bias_idx = (idx + i) % hidden_size;
            float val = reinterpret_cast<float*>(&vals)[i] + bias[bias_idx];
            
            // Fast GELU
            float val_cubed = val * val * val;
            float inner = sqrt_2_over_pi * (val + coeff * val_cubed);
            float tanh_val = tanhf(inner);
            
            reinterpret_cast<float*>(&vals)[i] = 0.5f * val * (1.0f + tanh_val);
        }
        
        *reinterpret_cast<float4*>(inout + idx) = vals;
    }
}


// ============================================================================
// C++ wrapper函数
// ============================================================================

torch::Tensor fused_ln_residual_optimized(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);
    const int total_tokens = batch_size * seq_len;
    
    if (hidden_size == 768) {
        const int threads = 256;
        const int blocks = total_tokens;
        
        fused_ln_residual_eval_kernel_768<<<blocks, threads>>>(
            input.data_ptr<float>(),
            residual.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            eps
        );
    }
    
    return output;
}


torch::Tensor fast_gelu(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int numel = input.numel();
    const int threads = 256;
    const int blocks = (numel / 4 + threads - 1) / threads;
    
    fast_gelu_kernel_vectorized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel
    );
    
    return output;
}


torch::Tensor optimized_softmax(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int batch_heads = input.size(0) * input.size(1);
    const int seq_len = input.size(2);
    
    if (seq_len == 128) {
        const int threads = 128;
        const int blocks = batch_heads * seq_len;
        
        online_softmax_kernel_128<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>()
        );
    } else {
        const int threads = 256;
        const int blocks = batch_heads * seq_len;
        
        online_softmax_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            seq_len
        );
    }
    
    return output;
}


void bias_gelu_fusion(torch::Tensor inout, torch::Tensor bias) {
    const int num_elements = inout.numel();
    const int hidden_size = bias.size(0);
    const int threads = 256;
    const int blocks = (num_elements / 4 + threads - 1) / threads;
    
    bias_gelu_fusion_kernel_vec<<<blocks, threads>>>(
        inout.data_ptr<float>(),
        bias.data_ptr<float>(),
        num_elements,
        hidden_size
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_ln_residual_optimized", &fused_ln_residual_optimized, 
          "Fused LayerNorm + Residual (optimized for eval)");
    m.def("fast_gelu", &fast_gelu, 
          "Fast GELU with tanh approximation");
    m.def("optimized_softmax", &optimized_softmax,
          "Optimized Softmax with online algorithm");
    m.def("bias_gelu_fusion", &bias_gelu_fusion,
          "Fused Bias addition and GELU activation");
}
