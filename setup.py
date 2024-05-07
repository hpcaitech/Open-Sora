from typing import List

from setuptools import find_packages, setup


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
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
