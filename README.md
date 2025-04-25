# Triton GPU programming for neural networks
## Introduction
Programming for accelerators such as GPUs is critical for modern AI systems. This often means programming directly in proprietary low-level languages such as CUDA. Triton is an alternative open-source language that allows you to code at a higher-level and compile to accelerators like GPU.

This lab is meant to teach you how to use Triton from first principles in an interactive fashion. You will start with trivial examples and build your way up to real algorithms like Flash Attention and Quantized neural networks. Through this hands-on experience, you will learn about basic GPU programming.

Please download the provided Jupyter Notebook files using the link below.
Follow the prompts and hints provided within the notebook to fill in the empty blocks and answer the questions.

[Part1](https://drive.google.com/file/d/1gfqmgNv0LgaiFbshBcUAS4DJ8BOabP8k/view) and [Part2](https://drive.google.com/file/d/1ZOnQE4e6_SHfRqJNnQUp9644SGW6Q77s/view) of the notebook are available for download.

## Environments
Before doing this lab, you will need to install following 4 libraries to run the code.
- Triton-Viz
```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/triton-viz.git
cd triton-viz
pip install -e .
```
- jaxtyping
```bash
pip install jaxtyping
```
- Pycairo
```bash
sudo apt update
sudo apt install -y libcairo2-dev pkg-config
pip install pycairo
```
- Matplotlib
```bash
pip install matplotlib
```

## Part 1: Trivial examples
In this part, you will start with trivial examples.

You will specifically learn about:

- The basic programming model of Triton.
- Pointer arithmetic.


## Part 2: Matrix Multiplication in Triton
In this part, you will write a very short high-performance FP16 matrix multiplication kernel.

You will specifically learn about:

- Block-level matrix multiplications.
- Multi-dimensional pointer arithmetic.
- Program re-ordering for improved L2 cache hit rate.