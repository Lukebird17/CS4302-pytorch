"""
自定义CUDA算子模块
"""
import torch

try:
    # 导入编译的CUDA扩展模块
    from custom_ops_cuda import (
        gemm, 
        gemm_bias, 
        gemm_bias_gelu, 
        layernorm,
        gemm_bias_add_layernorm,
        gemm_bias_gelu_add_layernorm
    )
    __all__ = [
        'gemm', 
        'gemm_bias', 
        'gemm_bias_gelu', 
        'layernorm',
        'gemm_bias_add_layernorm',
        'gemm_bias_gelu_add_layernorm'
    ]
except ImportError as e:
    import warnings
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    warnings.warn(
        f"无法加载自定义CUDA算子: {e}\n"
        f"请运行: cd {current_dir} && pip install -e . --no-build-isolation\n"
        f"如果仍然失败，请设置环境变量: export LD_LIBRARY_PATH=$(python3 -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),\"lib\"))'):$LD_LIBRARY_PATH",
        ImportWarning
    )

