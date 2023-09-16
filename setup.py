from setuptools import setup, find_packages

setup(
    name="snn_benchmark",
    version="1.0.0",
    description="A simple Python package for benchmarking SNNs using the SNN delays repository.",
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "dcls",
        "wandb",
        "pillow",
        "h5py",
        "tqdm",
        "scikit-learn",
        "fire"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)