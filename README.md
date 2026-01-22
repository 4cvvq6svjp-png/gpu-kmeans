# GPU K-Means Clustering

A high-performance implementation of the K-Means clustering algorithm using NVIDIA CUDA. This project aims to accelerate the clustering process by parallelizing distance calculations and centroid updates on the GPU.

## Features

* **GPU Acceleration**: Massively parallel computation of Euclidean distances.
* **Memory Optimization**: Utilizes Shared Memory to reduce global memory access latency.
* **CUDA Libraries**: Integrates `cuRAND` for random centroid initialization and `cuBLAS` for matrix operations.
* **Hybrid Architecture**: CPU handles orchestration while the GPU handles computationally intensive tasks.

## Prerequisites

To compile and run this project, you need:

* An NVIDIA GPU (CUDA compatible).
* CUDA Toolkit installed (verify with `nvcc --version`).
* GCC / G++ compiler.
* Make (optional, if using a Makefile).

## Installation and Compilation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:YourUsername/gpu-kmeans.git
    cd gpu-kmeans
    ```

2.  **Compile the project:**
    If you have a `Makefile`, simply run:
    ```bash
    make
    ```


## Usage

Run the generated binary:
modify the file name in the main.h before.
```bash
./kmeans /DATA-folder
