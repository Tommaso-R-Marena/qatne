from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qatne",
    version="1.0.0",
    author="Tommaso R. Marena",
    author_email="",
    description="Quantum Adaptive Tensor Network Eigensolver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tommaso-R-Marena/qatne",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "qiskit>=1.0.0",
        "qiskit-aer>=0.13.0",
        "qiskit-ibm-runtime>=0.15.0",
        "tensornetwork>=0.4.6",
        "openfermion>=1.5.0",
        "openfermionpyscf>=0.5",
        "pyscf>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "jupyterlab>=3.4.0",
        ],
    },
)
