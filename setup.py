import os
from setuptools import setup, find_packages

# Get the directory containing this file
here = os.path.abspath(os.path.dirname(__file__))

# Read the contents of the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SOFTS",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your dependencies here
        "scikit-learn",
        "numpy",
        "pandas",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here
        ],
    },
    author="carusyte",
    author_email="carusyte@163.com",
    description="Standalone package of https://github.com/Secilia-Cxy/SOFTS with engineering enhancements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carusyte/SOFTS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
