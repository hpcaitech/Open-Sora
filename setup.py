import os
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


def get_version() -> str:
    """
    This function reads the version.txt and generates the colossalai/version.py file.

    Returns:
        The library version stored in version.txt.
    """

    setup_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(setup_file_path)
    version_txt_path = os.path.join(project_path, "version.txt")

    with open(version_txt_path) as f:
        version = f.read().strip()
    return version


setup(
    name="open-sora",
    version=get_version(),
    packages=find_packages(
        exclude=(
            "docker",
            "tests",
            "docs",
            "examples",
            "tests",
            "scripts",
            "requirements",
            "extensions",
            "*.egg-info",
        ),
    ),
    description="Unofficial implementation of OpenAI's Sora by the Colossal-AI Team",
    long_description=fetch_readme(),
    long_description_content_type="text/markdown",
    license="Apache Software License 2.0",
    url="https://www.colossalai.org",
    project_urls={
        "Github": "https://github.com/hpcaitech/Open-Sora",
    },
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
