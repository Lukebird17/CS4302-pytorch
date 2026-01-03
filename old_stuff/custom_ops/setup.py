"""
编译脚本 - BERT 自定义算子（包含自实现GEMM）
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bert_custom_ops',
    ext_modules=[
        # 融合算子（LayerNorm, GELU等）
        CUDAExtension(
            name='bert_fused_ops',
            sources=['fused_ops.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                ]
            }
        ),
        # 自定义GEMM实现
        CUDAExtension(
            name='bert_custom_gemm',
            sources=['custom_gemm.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-fopenmp'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_75,code=sm_75',  # T4
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
                ]
            }
        ),
        CUDAExtension(
            name='bert_custom_transpose',
            sources=['custom_transpose.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--ptxas-options=-v',
                    # 包含常见的 GPU 架构优化
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_89,code=sm_89',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
