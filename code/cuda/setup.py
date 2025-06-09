from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cuda_kernels',
    ext_modules=[
        CUDAExtension(
            name='my_cuda_kernels',
            sources=[
                'kernels/base_kernel.cu',
                'kernels/coalesce_kernel.cu',
                'kernels/smem_base_kernel.cu',
                'kernels/smem_opt_kernel.cu'
            ],
            extra_compile_args={
                "nvcc": [
                    "-gencode=arch=compute_90,code=sm_90",
                    "-gencode=arch=compute_90,code=compute_90",
                ],
                # If you need to undefine Py_LIMITED_API for Python 3.12
                # "cxx": ["-U_Py_LIMITED_API"],
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)