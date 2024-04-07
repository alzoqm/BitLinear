# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup(
#     name='bitLinear',
#     ext_modules=[
#         CUDAExtension('bitLinear', [
#             'bit_cuda.cu',
#             'bitMatMul.cu',
#         ])
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bitLinear',
    ext_modules=[
        CUDAExtension('bitMatMul_extension', [
            'bitMatMul.cu',
        ]),
        CUDAExtension('bitCuda_extension', [
            'bit_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })