from typing import List
import importlib

from setuptools import find_packages, setup
import torch


def fetch_requirements(path) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def fetch_readme() -> str:
    """
    This function reads the README.md file in the current directory.

    Returns:
        The lines in the README file.
    """
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def is_npu_available():
    "Checks if `torch_npu` is installed and potentially if a NPU is in the environment"
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_npu") is None:
        return False

    import torch_npu

    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


if is_npu_available():
    requirements_file = "requirements_npu.txt"
else:
    requirements_file = "requirements.txt"

setup(
    name="opensora",
    version="1.1.0",
    packages=find_packages(
        exclude=(
            "assets",
            "configs",
            "docs",
            "outputs",
            "pretrained_models",
            "scripts",
            "tests",
            "tools",
            "*.egg-info",
        )
    ),
    description="Democratizing Efficient Video Production for All",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    install_requires=fetch_requirements(requirements_file),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
