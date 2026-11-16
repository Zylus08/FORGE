import os
import sys
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

PROJECT_ROOT = Path(__file__).resolve().parent

# Force C++20 and AVX2 on MSVC
extra_compile_args = ['/std:c++20', '/O2', '/arch:AVX2', '/EHsc']

setup(
    name='forge_native',
    ext_modules=[
        CppExtension(
            name='forge._forge_native',
            sources=[str(PROJECT_ROOT / 'native' / 'src' / 'fused_ops_m2.cpp')],
            include_dirs=[str(PROJECT_ROOT / 'native' / 'include')],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
