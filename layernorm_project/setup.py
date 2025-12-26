"""
自定义LayerNorm CUDA扩展的编译脚本
使用方法: python setup.py install
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='custom_layernorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='custom_layernorm_cuda',
            sources=['custom_layernorm.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    # 根据你的GPU架构选择合适的compute capability
                    # 常见的值：
                    # V100: sm_70
                    # RTX 2080/3080: sm_75/sm_86
                    # A100: sm_80
                    # H100: sm_90
                    '-gencode=arch=compute_70,code=sm_70',  # V100
                    '-gencode=arch=compute_75,code=sm_75',  # Turing
                    '-gencode=arch=compute_80,code=sm_80',  # A100
                    '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

