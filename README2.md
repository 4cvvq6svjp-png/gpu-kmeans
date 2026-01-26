# CUDA K-Means Implementation

**Authors:** Maximilien Lyonnais, Antoine Boubée
**Date:** 26/01/2026

## Overview
This repository contains a high-performance implementation of the K-means clustering algorithm using CUDA. The project is optimized for the NVIDIA GeForce 2080 Ti architecture.

## Data Structure & Memory Layout
To optimize memory access patterns, we utilized a **transposed array format** (Structure of Arrays - SoA).
* **Format:** Data is stored as $x_0, x_1, x_2, ..., y_0, y_1, y_2, ...$ rather than an Array of Structures ($x_0, y_0, x_1, y_1...$).
* **Coalescing:** This layout applies to both input instances and centroids. Storing data by dimension allows threads in a warp to access consecutive memory addresses, ensuring fully coalesced memory transactions.

![Memory Layout and Coalescing Diagram](assets/memory_layout.png)

## Implementation Details

### 1. Centroid Initialization (`kernel_InitializeCentroids`)
This kernel initializes the starting centroids by selecting $K$ random points from the input dataset.
* **Parallelism:** Each thread is responsible for initializing one centroid.
* **Random Generation:** Uses the `curand` library directly on the GPU. Threads generate a float (0.0 to 1.0) and convert it to an integer index.
* **Independence:** Each thread operates on its own memory zone, ensuring total independence between threads.

### 2. Label Assignment
Each thread corresponds to one data point and assigns it a label based on the nearest centroid.
* **Shared Memory:** Current centroid positions are loaded into shared memory to accelerate distance calculations.
* **Memory Access:** Coordinates are retrieved from global memory. Since $x$ values are contiguous, memory access remains coalesced.
* **Output:** New labels are written to `GPU_label` in a coalesced manner.

### 3. Centroid Update (Accumulation & Division)
The update process is split into two steps: Accumulation (summing coordinates) and Averaging.

#### Step A: Accumulation
This is the most computationally expensive step due to concurrent writes.
* **Challenge:** Direct use of `atomicAdd` on global memory would cause significant serialization and performance loss.
* **Solution:** We use **shared memory accumulators**. Each thread block has its own shared memory, reducing conflicts to the block level rather than the entire GPU.
* **Process:**
    1. Shared memory is initialized and cleaned.
    2. Threads increment the local cluster counter and add coordinates to the local sum in shared memory.
    3. Partial results are flushed to global memory after block processing.

#### Step B: Averaging
This kernel calculates the final mean position for the new centroids.
* **Execution:** Each thread handles one centroid, retrieves the point count, and divides the accumulated sums.
* **Performance:** Writes to `GPU_centroid_T` are coalesced as adjacent threads write to adjacent memory zones.

## Performance Results
The implementation was benchmarked on a GeForce 2080 Ti. The metrics are measured in Mega-points per second.

We observed the following performance behaviors for datasets of $10^4$, $10^5$, and $10^6$ points:

![Benchmark Graph GeForce 2080 Ti](assets/benchmark_results.png)
