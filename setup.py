"""
Cognitive Swarm Framework - Package Setup

Multi-Agent Coordination Framework for RL Research
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cognitive_swarm",
    version="2.0.0",
    author="Cognitive Swarm Research Team",
    description="A unified framework for scalable, robust, and safe multi-agent coordination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cognitive-swarm/cognitive-swarm",
    packages=find_packages(exclude=["tests", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "viz": [
            "matplotlib>=3.5",
            "tensorboard>=2.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "cognitive-train=scripts.train:main",
            "cognitive-demo=scripts.full_system_demo:main",
        ],
    },
)
