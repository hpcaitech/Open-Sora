from typing import List

from setuptools import find_packages, setup


def fetch_requirements(paths) -> List[str]:
    """
    This function reads the requirements file.

    Args:
        path (str): the path to the requirements file.

    Returns:
        The lines in the requirements file.
    """
    if not isinstance(paths, list):
        paths = [paths]
    requirements = []
    for path in paths:
        with open(path, "r") as fd:
            requirements += [r.strip() for r in fd.readlines()]
    return requirements


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
    version="1.2.0",
    packages=find_packages(
        exclude=(
            "assets",
            "cache",
            "configs",
            "docs",
            "eval",
            "evaluation_results",
            "gradio",
            "logs",
            "notebooks",
            "outputs",
            "pretrained_models",
            "samples",
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
    url="https://github.com/hpcaitech/Open-Sora",
    project_urls={
        "Bug Tracker": "https://github.com/hpcaitech/Open-Sora/issues",
        "Examples": "https://hpcaitech.github.io/Open-Sora/",
        "Documentation": "https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file",
        "Github": "https://github.com/hpcaitech/Open-Sora",
    },
    install_requires=fetch_requirements("requirements/requirements.txt"),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    extras_require={
        "data": fetch_requirements("requirements/requirements-data.txt"),
        "eval": fetch_requirements("requirements/requirements-eval.txt"),
        "vae": fetch_requirements("requirements/requirements-vae.txt"),
        "full": fetch_requirements(
            [
                "requirements/requirements-data.txt",
                "requirements/requirements-eval.txt",
            ]
        ),
    },
)
