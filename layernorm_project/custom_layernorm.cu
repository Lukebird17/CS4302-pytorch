/*
 * 自定义LayerNorm CUDA实现
 * 
 * 实现了优化的LayerNorm forward kernel
 * 优化策略：
 * 1. 向量化内存访问 (float4)
 * 2. Warp shuffle reduction
 * 3. 共享内存优化
 * 4. 循环展开
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// Warp-level reduction using shuffle
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Block-level reduction
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];  // 最多32个warp
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp内reduction
    val = warpReduceSum(val);
    
    // 每个warp的第一个线程写入shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    // 第一个warp对所有warp的结果进行reduction
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    
    return val;
}


/*
 * LayerNorm Forward Kernel - 基础版本
 * 
 * 每个block处理一个样本的一个位置(batch * seq_len)
 * block内的线程并行处理hidden_size维度
 * 
 * 参数:
 *   input: [batch_size, seq_len, hidden_size]
 *   gamma: [hidden_size] - 缩放参数
 *   beta: [hidden_size] - 偏移参数
 *   output: [batch_size, seq_len, hidden_size]
 *   mean: [batch_size, seq_len] - 保存的均值(用于反向传播)
 *   rstd: [batch_size, seq_len] - 保存的标准差倒数
 */
template <typename scalar_t>
__global__ void layernorm_forward_kernel_basic(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ beta,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ rstd,
    int hidden_size,
    float eps) {
    
    // 每个block处理一个(batch, seq)位置
    int batch_seq_idx = blockIdx.x;
    
    // 指向当前位置的输入和输出
    const scalar_t* input_ptr = input + batch_seq_idx * hidden_size;
    scalar_t* output_ptr = output + batch_seq_idx * hidden_size;
    
    // Step 1: 计算均值
    float sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum += static_cast<float>(input_ptr[i]);
    }
    sum = blockReduceSum(sum);
    
    // Broadcast均值
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / hidden_size;
        if (mean != nullptr) {
            mean[batch_seq_idx] = static_cast<scalar_t>(s_mean);
        }
    }
    __syncthreads();
    float mean_val = s_mean;
    
    // Step 2: 计算方差
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = static_cast<float>(input_ptr[i]) - mean_val;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSum(var_sum);
    
    // Broadcast标准差倒数
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        float variance = var_sum / hidden_size;
        s_rstd = rsqrtf(variance + eps);  // 1 / sqrt(var + eps)
        if (rstd != nullptr) {
            rstd[batch_seq_idx] = static_cast<scalar_t>(s_rstd);
        }
    }
    __syncthreads();
    float rstd_val = s_rstd;
    
    // Step 3: 归一化并应用仿射变换
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = (static_cast<float>(input_ptr[i]) - mean_val) * rstd_val;
        float out = normalized * static_cast<float>(gamma[i]) + static_cast<float>(beta[i]);
        output_ptr[i] = static_cast<scalar_t>(out);
    }
}


/*
 * LayerNorm Forward Kernel - 优化版本（向量化）
 * 
 * 使用float4进行向量化内存访问，提高带宽利用率
 * 要求hidden_size能被4整除
 */
__global__ void layernorm_forward_kernel_vectorized(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int hidden_size,
    float eps) {
    
    int batch_seq_idx = blockIdx.x;
    
    // 转换为float4指针进行向量化访问
    const float4* input_vec = reinterpret_cast<const float4*>(input + batch_seq_idx * hidden_size);
    float4* output_vec = reinterpret_cast<float4*>(output + batch_seq_idx * hidden_size);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);
    
    int vec_size = hidden_size / 4;
    
    // Step 1: 计算均值（向量化）
    float sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = input_vec[i];
        sum += val.x + val.y + val.z + val.w;
    }
    sum = blockReduceSum(sum);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        s_mean = sum / hidden_size;
        if (mean != nullptr) {
            mean[batch_seq_idx] = s_mean;
        }
    }
    __syncthreads();
    float mean_val = s_mean;
    
    // Step 2: 计算方差（向量化）
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = input_vec[i];
        float diff_x = val.x - mean_val;
        float diff_y = val.y - mean_val;
        float diff_z = val.z - mean_val;
        float diff_w = val.w - mean_val;
        var_sum += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
    }
    var_sum = blockReduceSum(var_sum);
    
    __shared__ float s_rstd;
    if (threadIdx.x == 0) {
        float variance = var_sum / hidden_size;
        s_rstd = rsqrtf(variance + eps);
        if (rstd != nullptr) {
            rstd[batch_seq_idx] = s_rstd;
        }
    }
    __syncthreads();
    float rstd_val = s_rstd;
    
    // Step 3: 归一化并应用仿射变换（向量化）
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 val = input_vec[i];
        float4 g = gamma_vec[i];
        float4 b = beta_vec[i];
        
        float4 out;
        out.x = ((val.x - mean_val) * rstd_val) * g.x + b.x;
        out.y = ((val.y - mean_val) * rstd_val) * g.y + b.y;
        out.z = ((val.z - mean_val) * rstd_val) * g.z + b.z;
        out.w = ((val.w - mean_val) * rstd_val) * g.w + b.w;
        
        output_vec[i] = out;
    }
}


/*
 * C++ wrapper函数
 */

// 基础版本
torch::Tensor layernorm_forward_cuda_basic(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);
    const int total_tokens = batch_size * seq_len;
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({batch_size, seq_len}, input.options());
    auto rstd = torch::empty({batch_size, seq_len}, input.options());
    
    // 启动配置
    const int threads = 256;
    const int blocks = total_tokens;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "layernorm_forward_cuda_basic", ([&] {
        layernorm_forward_kernel_basic<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            rstd.data_ptr<scalar_t>(),
            hidden_size,
            eps
        );
    }));
    
    return output;
}

// 优化版本（向量化）
torch::Tensor layernorm_forward_cuda_optimized(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Optimized version only supports float32");
    TORCH_CHECK(input.size(2) % 4 == 0, "hidden_size must be divisible by 4");
    
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);
    const int total_tokens = batch_size * seq_len;
    
    auto output = torch::empty_like(input);
    auto mean = torch::empty({batch_size, seq_len}, input.options());
    auto rstd = torch::empty({batch_size, seq_len}, input.options());
    
    // 启动配置
    const int threads = 256;
    const int blocks = total_tokens;
    
    layernorm_forward_kernel_vectorized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        hidden_size,
        eps
    );
    
    return output;
}


/*
 * Python绑定
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_basic", &layernorm_forward_cuda_basic, "LayerNorm forward (CUDA) - basic version");
    m.def("forward_optimized", &layernorm_forward_cuda_optimized, "LayerNorm forward (CUDA) - optimized version");
}

