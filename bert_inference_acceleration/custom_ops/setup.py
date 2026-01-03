from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops_cuda',  # 改名避免冲突
            sources=['custom_gemm.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_70',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '--use_fast_math',
                    '-maxrregcount=128',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

