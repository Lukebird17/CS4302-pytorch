#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// 定义分块大小，32x32 是典型的高性能配置
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// 优化后的转置内核
// 参考了 Caffe2 的实现逻辑：使用共享内存并添加 padding 避免 Bank Conflict
__global__ void transpose_optimized_kernel(float* odata, const float* idata, int width, int height) {
    // 分配共享内存，+1 是为了避免同步读写时的 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width_in = width;

    // 1. 将数据从全局内存合并读取到共享内存
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            // 使用 __ldg 指令（如果支持）或标准读取
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width_in + x];
        }
    }

    __syncthreads();

    // 2. 重新计算坐标进行转置后的合并写入
    // 修改读取出的 tile 索引，实现转置
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 转置后的 x
    y = blockIdx.x * TILE_DIM + threadIdx.y;  // 转置后的 y
    int width_out = height;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * width_out + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// C++ 接口函数，用于被 Python 调用
torch::Tensor custom_transpose(torch::Tensor input) {
    auto device = input.device();
    int height = input.size(0);
    int width = input.size(1);
    
    auto output = torch::empty({width, height}, input.options());

    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    transpose_optimized_kernel<<<dimGrid, dimBlock>>>(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        width,
        height
    );

    return output;
}

// 绑定到 PyTorch 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose", &custom_transpose, "Custom optimized transpose (CUDA)");
}