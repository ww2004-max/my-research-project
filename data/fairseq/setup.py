#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


def write_version_py():
    version = "0.10.2"  # 手动指定一个版本号
    with open(os.path.join("fairseq", "version.py"), "w") as f:
        f.write(f'__version__ = "{version}"\n')
    return version


version = write_version_py()


with open("README.md", encoding="utf-8") as f:
    readme = f.read()


extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        "fairseq.libbleu",
        sources=[
            "fairseq/clib/libbleu/libbleu.cpp",
            "fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.token_block_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

# C++ Extensions
extensions.extend([
    cpp_extension.CppExtension("fairseq.libbase", sources=["fairseq/clib/libbase/balanced_assignment.cpp"]),
    cpp_extension.CppExtension("fairseq.libnat", sources=["fairseq/clib/libnat/edit_dist.cpp"]),
])

if "CUDA_HOME" in os.environ:
    extensions.extend([
        cpp_extension.CppExtension(
            "fairseq.libnat_cuda",
            sources=[
                "fairseq/clib/libnat_cuda/edit_dist.cu",
                "fairseq/clib/libnat_cuda/binding.cpp"
            ]
        ),
        cpp_extension.CppExtension(
            "fairseq.ngram_repeat_block_cuda",
            sources=[
                "fairseq/clib/cuda/ngram_repeat_block_cuda.cpp",
                "fairseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu"
            ]
        ),
    ])


cmdclass = {"build_ext": cpp_extension.BuildExtension}


def do_setup():
    setup(
        name="fairseq",
        version=version,
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=[
            "cffi",
            "cython",
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "torch>=1.13",
            "tqdm",
            "bitarray",
            "torchaudio>=0.8.0",
            "scikit-learn",
            "packaging",
        ],
        extras_require={
            "dev": ["flake8", "pytest", "black==22.3.0"],
            "docs": ["sphinx", "sphinx-argparse"],
        },
        packages=find_packages(
            exclude=[
                "examples",
                "scripts",
                "tests",
            ]
        ),
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-hydra-train = fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


if __name__ == "__main__":
    do_setup()