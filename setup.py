import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
from os import path

here = path.abspath(path.dirname(__file__))

# Use correct conda compiler used to build pytorch
os.environ['CXX'] = os.environ.get('GXX', '')

setup(
    name='focalpose',
    version='1.0.0',
    description='FocalPose',
    packages=find_packages(),
    ext_modules=[
        # CppExtension(
        #     name='cosypose_cext',
        #     sources=[
        #         'focalpose/csrc/cosypose_cext.cpp'
        #     ],
        #     extra_compile_args=['-O3'],
        #     verbose=True
        # )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
