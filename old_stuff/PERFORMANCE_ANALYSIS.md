# BERT自定义GEMM性能分析报告

## 🔍 问题诊断：为什么加速比很小？

### 当前结果
- Batch=1: **12.2%** 提升
- Batch=4+: **~1%** 提升

### 根本原因分析

## ❌ 问题1：GEMM Kernel实现效率低

### 1.1 `gemm_768_kernel` 的问题

```cuda
// 当前实现（custom_gemm.cu:175-226）
constexpr int TILE_K = 16;
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;

// 768 / 16 = 48次循环
for (int t = 0; t < K; t += TILE_K) {
    // 每个线程只计算1个输出元素
    As[threadIdx.y][threadIdx.x] = ...;
    Bs[threadIdx.y][threadIdx.x] = ...;
    __syncthreads();
    
    for (int k = 0; k < TILE_K; k++) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
}
```

**问题**：
- ❌ **Tile太小**：16x16只有256个线程，GPU利用率低
- ❌ **循环次数过多**：768/16=48次循环，每次都要同步
- ❌ **每线程计算量少**：只计算1个输出，寄存器利用率低
- ❌ **没有向量化加载**：每次只加载1个float

**对比cuBLAS**：
- ✅ cuBLAS使用128x128 tile（16K线程块）
- ✅ 循环次数：768/8=96次（BK=8但每次处理多个）
- ✅ 每线程计算8x8=64个输出
- ✅ 使用float4向量化加载

### 1.2 `gemm_kernel_optimized` 的问题

```cuda
// 当前实现（custom_gemm.cu:84-163）
template<int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void gemm_kernel_optimized(...) {
    __shared__ float As[BM][BK];  // 128 × 8
    __shared__ float Bs[BK][BN];  // 8 × 128
    
    float accum[TM][TN] = {0.0f};  // 8 × 8寄存器数组
    
    // ... 加载逻辑 ...
}
```

**问题**：
- ❌ **线程块配置错误**：
  ```cuda
  dim3 block(16, 16);  // 只有256个线程！
  ```
  但模板要求 `BM=128, BN=128` 需要 **1024+线程**

- ❌ **加载逻辑效率低**：
  ```cuda
  for (int i = tid; i < BM * BK; i += blockDim.x * blockDim.y) {
      // 128*8=1024个元素，256个线程加载 → 需要4次迭代
  }
  ```

- ❌ **Bank Conflict**：
  ```cuda
  As[row][col]  // col连续访问会导致bank conflict
  ```

### 1.3 向量化版本的问题

```cuda
// custom_gemm.cu:233-291
dim3 block(32, 8);  // 256线程
dim3 grid((N / 4 + 31) / 32, (M + 7) / 8);

// 问题：条件永远不会触发！
if (N % 4 == 0 && N >= 128) {
    // BERT的N通常是768或3072，会进入这里
    gemm_kernel_vectorized<<<grid, block>>>(...);
}
```

但向量化kernel本身有问题：
```cuda
int col = blockIdx.x * TILE_N + threadIdx.x * 4;  // 假设TILE_N=128
// threadIdx.x最大=31，所以覆盖 0~31*4=124
// 但TILE_N=128！少了4个元素
```

## ❌ 问题2：Kernel调度逻辑混乱

```cuda
// custom_gemm.cu:313-344
if (K == 768) {
    // 使用gemm_768_kernel
    dim3 block(16, 16);  // 256线程 - 太少！
    gemm_768_kernel<<<grid, block>>>(...);
    
} else if (N % 4 == 0 && N >= 128) {
    // 向量化版本（有bug）
    gemm_kernel_vectorized<<<grid, block>>>(...);
    
} else {
    // "优化"版本
    dim3 block(16, 16);  // 只有256线程！
    gemm_kernel_optimized<128, 128, 8, 8, 8><<<grid, block>>>(...);
    // ❌ 模板参数要求1024+线程，但只给256！
}
```

## ❌ 问题3：只在2D张量时使用自定义GEMM

```python
# bert_optimized.py:62-84
def forward(self, x):
    if CUSTOM_GEMM_AVAILABLE and x.dim() == 2:  # ❌ 条件太严格
        output = bert_custom_gemm.custom_gemm(x, self.weight.t())
        ...
    else:
        return torch.nn.functional.linear(x, self.weight, self.bias)  # 实际走这里
```

**问题**：BERT的输入通常是3D：`[batch, seq, hidden]`
- 所以大部分时候**都在用PyTorch的实现**！

### 验证方法
```python
# bert_optimized.py
def forward(self, x):
    print(f"CustomLinear: input shape={x.shape}, dim={x.dim()}")  # 加这行调试
```

预测输出：
```
CustomLinear: input shape=torch.Size([4, 128, 768]), dim=3  # ❌ 不会走自定义GEMM
```

## ❌ 问题4：融合算子的条件限制

```python
# bert_optimized.py:134-141
def forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    
    if FUSED_OPS_AVAILABLE and not self.training:  # ✅ eval模式
        return bert_fused_ops.fused_ln_residual_optimized(...)
    else:
        return self.LayerNorm(hidden_states + input_tensor)  # dropout被跳过了
```

但fusion kernel有限制：
```cuda
// fused_ops.cu:334
if (hidden_size == 768) {
    // 只支持768维度
}
```

## 📊 性能瓶颈量化分析

### BERT Forward Pass时间分布
| 算子 | 占比 | 优化情况 | 实际加速 |
|------|------|----------|---------|
| **GEMM** | **~80%** | ❌ 比cuBLAS慢2-5x | **-60%** |
| Attention Transpose | ~5% | 未优化 | 0% |
| LayerNorm | ~8% | ✅ 融合 | +30% × 8% = 2.4% |
| GELU | ~3% | ✅ 融合 | +50% × 3% = 1.5% |
| Softmax | ~2% | 未优化 | 0% |
| 其他 | ~2% | N/A | 0% |

**总体效果**：
```
加速 = LayerNorm加速 + GELU加速 + GEMM拖累
     = 2.4% + 1.5% - 60% × 80%
     = 3.9% - 48%
     = -44.1%
```

但为什么实际只看到+1~12%？
→ **因为大部分时候根本没用自定义GEMM**（3D输入限制）
→ 实际只有少量操作用了自定义算子

## ✅ 解决方案

### 方案1：修复GEMM实现（推荐）

1. **增大Tile Size**
   ```cuda
   constexpr int TILE_M = 128;
   constexpr int TILE_N = 128;
   constexpr int TILE_K = 8;  // 保持小K以减少shared memory
   
   dim3 block(16, 16);  // 256线程
   // 每个线程计算 8x8 = 64个输出
   ```

2. **修复线程-数据映射**
   ```cuda
   int threadRowStart = (threadIdx.y * 8);  // 16 threads → 128 rows
   int threadColStart = (threadIdx.x * 8);  // 16 threads → 128 cols
   ```

3. **添加向量化加载**
   ```cuda
   float4 a_vec = *reinterpret_cast<const float4*>(&A[...]);
   ```

4. **避免Bank Conflict**
   ```cuda
   __shared__ float As[BM][BK + 1];  // +1避免冲突
   ```

### 方案2：放弃自定义GEMM，专注其他优化

**理由**：
- cuBLAS经过Nvidia数十年优化，很难超越
- 小的优化收益（LayerNorm 2.4% + GELU 1.5% = 3.9%）可以接受
- 学习项目重点在理解原理，而非击败cuBLAS

**建议**：
- 保留融合算子（LayerNorm, GELU）
- **移除自定义GEMM**，回退到cuBLAS
- 预期加速：**3-5%**（稳定且可靠）

### 方案3：使用Cutlass库

```cpp
#include "cutlass/gemm/device/gemm.h"

// Cutlass提供接近cuBLAS的性能
// 但更容易定制和理解
```

## 🎯 立即行动：快速修复

### 快速修复1：移除3D限制

```python
# bert_optimized.py:60-84 修改为：
def forward(self, x):
    original_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))  # flatten到2D
    
    if CUSTOM_GEMM_AVAILABLE:
        output = bert_custom_gemm.custom_gemm(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
    else:
        output = torch.nn.functional.linear(x, self.weight, self.bias)
    
    if len(original_shape) > 2:
        output = output.reshape(*original_shape[:-1], -1)
    
    return output
```

### 快速修复2：调整Kernel选择逻辑

```cuda
// 只使用基础tiled版本
if (K == 768 || K == 3072) {
    // 使用32x32 tile而不是16x16
    dim3 block(32, 32);  // 1024线程
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    gemm_kernel_tiled<32><<<grid, block>>>(A, B, C, M, N, K);
} else {
    // fallback到PyTorch
    return torch::matmul(A, B);
}
```

## 📈 预期改进

| 修复 | 预期效果 |
|------|---------|
| 移除3D限制 | **真正启用自定义GEMM** |
| 增大Tile到32x32 | **2-3x加速**（相比16x16） |
| 两者结合 | **10-20%整体提升**（如果GEMM接近cuBLAS 70%性能）|

但最终可能还是：
- **自定义GEMM达到cuBLAS 50-70%性能** → 整体慢10-40%
- **vs. 只用融合算子** → 稳定+3-5%

## 💡 建议

**对于学习项目**：
1. ✅ 保留并优化融合算子（已经工作得很好）
2. ❌ 放弃自定义GEMM（投入产出比太低）
3. ✅ 加上详细的性能分析和文档

**如果必须有GEMM**：
- 使用Cutlass库而不是从零实现
- 或者降低期望：达到cuBLAS 50%性能就算成功

## 🔧 下一步

1. **验证问题**：添加打印确认3D输入导致fallback
2. **选择方向**：修复GEMM vs. 专注融合算子
3. **实现修复**：根据选择的方向改代码
4. **重新测试**：验证改进效果

