import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
readme_path = HERE / "README.md"
README = (
    readme_path.read_text(encoding="utf-8")
    if readme_path.exists()
    else "Dynamic warehouse environment and LLM-augmented MAPPO research code."
)

setup(
    name="llm-mappo",
    version="0.1.0",
    description="Dynamic warehouse environment and LLM-augmented MAPPO research code",
    long_description=README,
    long_description_content_type="text/markdown",
    author="LLM-MAPPO contributors",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    package_data={
        "rware.layouts.o3": ["*.txt"],
        "llm_mappo": ["data/formal_environment.json"],
    },
    classifiers=[
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "numpy",
        "gymnasium",
        "pyglet<2",
        "networkx",
    ],
    python_requires=">=3.10,<3.11",
    extras_require={
        "test": ["pytest"],
        "dev": ["build", "flake8", "pytest", "PyYAML", "scipy", "tensorboard"],
        "train": ["torch", "scipy", "matplotlib"],
    },
    include_package_data=True,
)
