from setuptools import setup, find_packages
import os

setup(
    name="thermomaps",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Lukas",
    author_email="lherron@umd.edu",
    description="A Python module to study molecular dynamics simulations using Thermodynamic Maps",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=[f'cli/ising_cli.py' , f'cli/tm_cli.py'],
    entry_points={
        'console_scripts': [
            'ising_model = ising_cli:main',
            'thermomaps = tm_cli:main',
        ],
    },
)

