#!/usr/bin/env python
from setuptools import setup

setup(
    name="ReverseEncodingTree",
    version="1.2.2",
    description="library for the Reverse Encoding Tree",
    long_description="NeuroEvolution is one of the most competitive evolutionary learning strategies for "
                     "designing novel neural networks for use in specific tasks. "
                     "This library implemented an evolutionary strategy named Reverse Encoding Tree (RET), "
                     "and expanded this strategy to evolve neural networks (Bi-NEAT and GS-NEAT). "
                     "The experiments of RET contain the landscapes of Mount Everest and Rastrigin Function, and  "
                     "those of RET-based NEAT include logic gates, Cartpole V0, and Lunar Lander V2.",
    author="Haoling Zhang, Chao-Han Huck Yang",
    author_email="zhanghaoling@genomics.cn",
    url="https://github.com/HaolingZHANG/ReverseEncodingTree",
    packages=[
        "ReverseEncodingTree",
        "ReverseEncodingTree.benchmark",
        "ReverseEncodingTree.benchmark.dataset",
        "ReverseEncodingTree.benchmark.methods",
        "ReverseEncodingTree.benchmark.results",
        "ReverseEncodingTree.configures",
        "ReverseEncodingTree.configures.example",
        "ReverseEncodingTree.configures.task",
        "ReverseEncodingTree.evolution",
        "ReverseEncodingTree.evolution.bean",
        "ReverseEncodingTree.evolution.methods",
        "ReverseEncodingTree.example",
        "ReverseEncodingTree.output",
        "ReverseEncodingTree.tasks",
        "ReverseEncodingTree.utils",
    ],
    package_data={
        "ReverseEncodingTree": [
            "benchmark/dataset/mount_everest.csv",
            "benchmark/dataset/rastrigin.csv",
            "configures/example/cart-pole-v0",
            "configures/example/lunar-lander-v2",
            "configures/example/xor",
            "configures/task/cart-pole-v0.bi",
            "configures/task/cart-pole-v0.fs",
            "configures/task/cart-pole-v0.gs",
            "configures/task/cart-pole-v0.n",
            "configures/task/logic.bi",
            "configures/task/logic.fs",
            "configures/task/logic.gs",
            "configures/task/logic.n",
            "configures/task/lunar-lander-v2.bi",
            "configures/task/lunar-lander-v2.fs",
            "configures/task/lunar-lander-v2.gs",
            "configures/task/lunar-lander-v2.n",
        ]
    },
    package_dir={"ReverseEncodingTree": "."},
    install_requires=[
        "numpy", "matplotlib", "graphviz", "neat-python", "sklearn", "gym", "six", "pandas"
    ],
    license="Apache",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="Evolutionary Strategy, NeuroEvolution",
)
