from setuptools import setup, find_packages

setup(
    name="snn_benchmark",
    version="1.0.0",
    description="A simple Python package for benchmarking SNNs using the SNN delays repository.",
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        "torch",
        "torchaudio",
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
        "fire",
    ],
)