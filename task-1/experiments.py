"""
GPU-Accelerated Vector Search Experiments

This script measures performance of GPU-accelerated distance functions and KMeans implementations
against CPU-based implementations across varying data sizes and dimensions.

Tests include:
1. Distance function scaling with vector counts (4,000 to 4,000,000)
2. KMeans performance with varying vector counts (4,000 to 500,000)
3. KMeans performance with varying dimensions (2D to 32,768D)
"""

import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from task import *

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Test parameters
DISTANCE_VECTOR_COUNTS = [4000, 10000, 50000, 100000, 500000, 1000000, 4000000]
KMEANS_VECTOR_COUNTS = [4000, 10000, 20000, 50000, 100000]
VECTOR_DIMENSIONS = [2, 8, 32, 128, 512, 2048, 8192, 32768]
KMEANS_CLUSTERS = 10
FIXED_DIMENSION = 128
FIXED_VECTORS = 10000
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# CPU Implementation of Distance Functions
# -----------------------------------------------------------------------------

def cpu_l2_distance(A, X):
    """Compute L2 distances between each vector in A and X using CPU."""
    return np.sqrt(np.sum((A - X) ** 2, axis=1))

def cpu_cosine_distance(A, X):
    """Compute Cosine distances between each vector in A and X using CPU."""
    dot_products = np.sum(A * X, axis=1)
    A_norms = np.sqrt(np.sum(A ** 2, axis=1))
    X_norm = np.sqrt(np.sum(X ** 2))
    return 1.0 - dot_products / (A_norms * X_norm)

def cpu_dot_distance(A, X):
    """Compute negative dot product between each vector in A and X using CPU."""
    return -np.sum(A * X, axis=1)

def cpu_manhattan_distance(A, X):
    """Compute Manhattan distances between each vector in A and X using CPU."""
    return np.sum(np.abs(A - X), axis=1)

# -----------------------------------------------------------------------------
# CPU Implementation of KMeans
# -----------------------------------------------------------------------------

def cpu_kmeans_plus_plus_l2(A, K):
    """KMeans++ initialization using L2 distance (CPU version)."""
    N = A.shape[0]
    centroids = []
    
    # Randomly select the first centroid
    first_idx = np.random.randint(0, N)
    centroids.append(A[first_idx])
    
    for _ in range(1, K):
        # Compute distance from each point to nearest centroid
        distances = cpu_l2_distance(A, centroids[0]) ** 2
        
        for c in centroids[1:]:
            d_new = cpu_l2_distance(A, c) ** 2
            distances = np.minimum(distances, d_new)
        
        # Select next centroid proportional to distances
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        next_idx = np.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx])
    
    return np.array(centroids)

def cpu_kmeans_l2(N, D, A, K):
    """KMeans clustering using L2 distance (CPU version)."""
    centroids = cpu_kmeans_plus_plus_l2(A, K)
    
    max_iter = 100
    tol = 1e-4
    
    for i in range(max_iter):
        # Compute distances to all centroids
        distances = np.zeros((N, K))
        
        for j in range(K):
            distances[:, j] = cpu_l2_distance(A, centroids[j])
        
        # Assign points to nearest centroid
        cluster_ids = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[j])
        
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
            
        centroids = new_centroids
    
    return cluster_ids.tolist()

def cpu_kmeans_plus_plus_cosine(A, K):
    """KMeans++ initialization using Cosine distance (CPU version)."""
    N = A.shape[0]
    centroids = []
    
    # Randomly select the first centroid
    first_idx = np.random.randint(0, N)
    centroids.append(A[first_idx])
    
    for _ in range(1, K):
        # Compute distance from each point to nearest centroid
        distances = cpu_cosine_distance(A, centroids[0]) ** 2
        
        for c in centroids[1:]:
            d_new = cpu_cosine_distance(A, c) ** 2
            distances = np.minimum(distances, d_new)
        
        # Select next centroid proportional to distances
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        next_idx = np.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx])
    
    return np.array(centroids)

def cpu_kmeans_cosine(N, D, A, K):
    """KMeans clustering using Cosine distance (CPU version)."""
    centroids = cpu_kmeans_plus_plus_cosine(A, K)
    
    max_iter = 100
    tol = 1e-4
    
    for i in range(max_iter):
        # Compute distances to all centroids
        distances = np.zeros((N, K))
        
        for j in range(K):
            distances[:, j] = cpu_cosine_distance(A, centroids[j])
        
        # Assign points to nearest centroid
        cluster_ids = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[j])
        
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
            
        centroids = new_centroids
    
    return cluster_ids.tolist()

def cpu_kmeans_plus_plus_dot(A, K):
    """KMeans++ initialization using negative dot product (CPU version)."""
    N = A.shape[0]
    centroids = []
    
    # Randomly select the first centroid
    first_idx = np.random.randint(0, N)
    centroids.append(A[first_idx])
    
    for _ in range(1, K):
        # Compute distance from each point to nearest centroid
        distances = cpu_dot_distance(A, centroids[0]) ** 2
        
        for c in centroids[1:]:
            d_new = cpu_dot_distance(A, c) ** 2
            distances = np.minimum(distances, d_new)
        
        # Select next centroid proportional to distances
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        next_idx = np.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx])
    
    return np.array(centroids)

def cpu_kmeans_dot(N, D, A, K):
    """KMeans clustering using negative dot product (CPU version)."""
    centroids = cpu_kmeans_plus_plus_dot(A, K)
    
    max_iter = 100
    tol = 1e-4
    
    for i in range(max_iter):
        # Compute distances to all centroids
        distances = np.zeros((N, K))
        
        for j in range(K):
            distances[:, j] = cpu_dot_distance(A, centroids[j])
        
        # Assign points to nearest centroid
        cluster_ids = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[j])
        
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
            
        centroids = new_centroids
    
    return cluster_ids.tolist()

def cpu_kmeans_plus_plus_manhattan(A, K):
    """KMeans++ initialization using Manhattan distance (CPU version)."""
    N = A.shape[0]
    centroids = []
    
    # Randomly select the first centroid
    first_idx = np.random.randint(0, N)
    centroids.append(A[first_idx])
    
    for _ in range(1, K):
        # Compute distance from each point to nearest centroid
        distances = cpu_manhattan_distance(A, centroids[0]) ** 2
        
        for c in centroids[1:]:
            d_new = cpu_manhattan_distance(A, c) ** 2
            distances = np.minimum(distances, d_new)
        
        # Select next centroid proportional to distances
        probs = distances / distances.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.random()
        next_idx = np.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx])
    
    return np.array(centroids)

def cpu_kmeans_manhattan(N, D, A, K):
    """KMeans clustering using Manhattan distance (CPU version)."""
    centroids = cpu_kmeans_plus_plus_manhattan(A, K)
    
    max_iter = 100
    tol = 1e-4
    
    for i in range(max_iter):
        # Compute distances to all centroids
        distances = np.zeros((N, K))
        
        for j in range(K):
            distances[:, j] = cpu_manhattan_distance(A, centroids[j])
        
        # Assign points to nearest centroid
        cluster_ids = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.shape[0] > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[j])
        
        new_centroids = np.array(new_centroids)
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
            
        centroids = new_centroids
    
    return cluster_ids.tolist()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def generate_dataset(N, D, seed=RANDOM_SEED):
    """Generate a random dataset with fixed seed for reproducibility."""
    np.random.seed(seed)
    return np.random.randn(N, D).astype(np.float32)

def plot_and_save(results, plot_title, filename, x_key, is_dimension=False):
    """Create standardized plots and save them to file."""
    plt.figure(figsize=(12, 8))
    
    # X-axis label
    x_label = 'Vector Dimension' if is_dimension else 'Number of Vectors'
    
    # Plot GPU execution times
    plt.subplot(2, 2, 1)
    plt.loglog(results[x_key], results['l2_gpu_time'], 'o-', label='L2')
    plt.loglog(results[x_key], results['cosine_gpu_time'], 's-', label='Cosine')
    plt.loglog(results[x_key], results['dot_gpu_time'], '^-', label='Dot Product')
    plt.loglog(results[x_key], results['manhattan_gpu_time'], 'D-', label='Manhattan')
    plt.grid(True, which="both", ls="--")
    plt.xlabel(x_label)
    plt.ylabel('GPU Execution Time (s)')
    plt.title(f'{plot_title} - GPU Execution Time')
    plt.legend()
    
    # Plot CPU execution times
    plt.subplot(2, 2, 2)
    plt.loglog(results[x_key], results['l2_cpu_time'], 'o-', label='L2')
    plt.loglog(results[x_key], results['cosine_cpu_time'], 's-', label='Cosine')
    plt.loglog(results[x_key], results['dot_cpu_time'], '^-', label='Dot Product')
    plt.loglog(results[x_key], results['manhattan_cpu_time'], 'D-', label='Manhattan')
    plt.grid(True, which="both", ls="--")
    plt.xlabel(x_label)
    plt.ylabel('CPU Execution Time (s)')
    plt.title(f'{plot_title} - CPU Execution Time')
    plt.legend()
    
    # Plot speedup
    plt.subplot(2, 2, 3)
    plt.semilogx(results[x_key], results['l2_speedup'], 'o-', label='L2')
    plt.semilogx(results[x_key], results['cosine_speedup'], 's-', label='Cosine')
    plt.semilogx(results[x_key], results['dot_speedup'], '^-', label='Dot Product')
    plt.semilogx(results[x_key], results['manhattan_speedup'], 'D-', label='Manhattan')
    plt.grid(True, which="both", ls="--")
    plt.xlabel(x_label)
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title(f'{plot_title} - GPU Speedup')
    plt.legend()
    
    # Plot metric comparison on GPU
    plt.subplot(2, 2, 4)
    width = 0.2
    
    # Choose subset of data points for bar chart to avoid overcrowding
    if len(results[x_key]) > 5:
        indices = list(range(0, len(results[x_key]), len(results[x_key]) // 5))
        indices = sorted(set(indices + [len(results[x_key]) - 1]))  # Ensure last point is included
        
        x_values = [results[x_key][i] for i in indices]
        l2_vals = [results['l2_speedup'][i] for i in indices]
        cos_vals = [results['cosine_speedup'][i] for i in indices]
        dot_vals = [results['dot_speedup'][i] for i in indices]
        man_vals = [results['manhattan_speedup'][i] for i in indices]
    else:
        x_values = results[x_key]
        l2_vals = results['l2_speedup']
        cos_vals = results['cosine_speedup']
        dot_vals = results['dot_speedup']
        man_vals = results['manhattan_speedup']
    
    x = np.arange(len(x_values))
    plt.bar(x - width*1.5, l2_vals, width, label='L2')
    plt.bar(x - width*0.5, cos_vals, width, label='Cosine')
    plt.bar(x + width*0.5, dot_vals, width, label='Dot Product')
    plt.bar(x + width*1.5, man_vals, width, label='Manhattan')
    
    plt.xticks(x, [str(val) for val in x_values], rotation=45)
    plt.xlabel(x_label)
    plt.ylabel('Speedup (CPU Time / GPU Time)')
    plt.title(f'{plot_title} - Speedup by Distance Metric')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'figures/{filename}.png')

# -----------------------------------------------------------------------------
# Experiment Functions
# -----------------------------------------------------------------------------

def test_distance_functions_scaling():
    """Test how distance functions scale with increasing vector counts."""
    print("Testing distance functions scaling with vector numbers...")
    
    # Results storage
    results = {
        'vector_count': [],
        'l2_gpu_time': [], 'cosine_gpu_time': [], 'dot_gpu_time': [], 'manhattan_gpu_time': [],
        'l2_cpu_time': [], 'cosine_cpu_time': [], 'dot_cpu_time': [], 'manhattan_cpu_time': [],
        'l2_speedup': [], 'cosine_speedup': [], 'dot_speedup': [], 'manhattan_speedup': []
    }
    
    for N in DISTANCE_VECTOR_COUNTS:
        print(f"  Testing with {N} vectors...")
        
        # Generate dataset
        A_np = generate_dataset(N, FIXED_DIMENSION)
        X_np = generate_dataset(1, FIXED_DIMENSION)[0]  # Single query vector
        
        # Convert to GPU tensors
        A = torch.tensor(A_np, device="cuda", dtype=torch.float32)
        X = torch.tensor(X_np, device="cuda", dtype=torch.float32)
        
        # Setup
        distances = torch.empty(N, device="cuda", dtype=torch.float32)
        grid = (N,)
        block_size = 256 if FIXED_DIMENSION > 512 else 128
        
        # Test L2 distance
        # Warm-up
        l2_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        # Timing
        torch.cuda.synchronize()
        start_time = time.time()
        l2_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        l2_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_l2_distance(A_np, X_np)
        l2_cpu_time = time.time() - start_time
        
        # Test Cosine distance
        cosine_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        start_time = time.time()
        cosine_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        cosine_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_cosine_distance(A_np, X_np)
        cosine_cpu_time = time.time() - start_time
        
        # Test Dot product distance
        dot_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        start_time = time.time()
        dot_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        dot_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_dot_distance(A_np, X_np)
        dot_cpu_time = time.time() - start_time
        
        # Test Manhattan distance
        manhattan_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        start_time = time.time()
        manhattan_distance_kernel[grid](A, X, distances, FIXED_DIMENSION, A.stride(0), BLOCK_SIZE=block_size)
        torch.cuda.synchronize()
        manhattan_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_manhattan_distance(A_np, X_np)
        manhattan_cpu_time = time.time() - start_time
        
        # Calculate speedups
        l2_speedup = l2_cpu_time / l2_gpu_time
        cosine_speedup = cosine_cpu_time / cosine_gpu_time
        dot_speedup = dot_cpu_time / dot_gpu_time
        manhattan_speedup = manhattan_cpu_time / manhattan_gpu_time
        
        # Store results
        results['vector_count'].append(N)
        results['l2_gpu_time'].append(l2_gpu_time)
        results['cosine_gpu_time'].append(cosine_gpu_time)
        results['dot_gpu_time'].append(dot_gpu_time)
        results['manhattan_gpu_time'].append(manhattan_gpu_time)
        results['l2_cpu_time'].append(l2_cpu_time)
        results['cosine_cpu_time'].append(cosine_cpu_time)
        results['dot_cpu_time'].append(dot_cpu_time)
        results['manhattan_cpu_time'].append(manhattan_cpu_time)
        results['l2_speedup'].append(l2_speedup)
        results['cosine_speedup'].append(cosine_speedup)
        results['dot_speedup'].append(dot_speedup)
        results['manhattan_speedup'].append(manhattan_speedup)
        
        # Print results
        print(f"    L2 Distance - GPU: {l2_gpu_time:.6f}s, CPU: {l2_cpu_time:.6f}s, Speedup: {l2_speedup:.2f}x")
        print(f"    Cosine Distance - GPU: {cosine_gpu_time:.6f}s, CPU: {cosine_cpu_time:.6f}s, Speedup: {cosine_speedup:.2f}x")
        print(f"    Dot Product - GPU: {dot_gpu_time:.6f}s, CPU: {dot_cpu_time:.6f}s, Speedup: {dot_speedup:.2f}x")
        print(f"    Manhattan Distance - GPU: {manhattan_gpu_time:.6f}s, CPU: {manhattan_cpu_time:.6f}s, Speedup: {manhattan_speedup:.2f}x")
    
    # Save and plot results
    pd.DataFrame(results).to_csv('results/distance_functions_scaling.csv', index=False)
    plot_and_save(results, 'Distance Functions', 'distance_functions_scaling', 'vector_count')
    
    print("Distance functions scaling test completed.")

def test_kmeans_vector_count():
    """Test KMeans performance with varying vector counts."""
    print("Testing KMeans with varying vector counts...")
    
    # Results storage
    results = {
        'vector_count': [],
        'l2_gpu_time': [], 'cosine_gpu_time': [], 'dot_gpu_time': [], 'manhattan_gpu_time': [],
        'l2_cpu_time': [], 'cosine_cpu_time': [], 'dot_cpu_time': [], 'manhattan_cpu_time': [],
        'l2_speedup': [], 'cosine_speedup': [], 'dot_speedup': [], 'manhattan_speedup': []
    }
    
    for N in KMEANS_VECTOR_COUNTS:
        print(f"  Testing with {N} vectors...")
        
        # Generate dataset
        A_np = generate_dataset(N, FIXED_DIMENSION)
        
        # Test L2 KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_l2(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        l2_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_l2(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        l2_cpu_time = time.time() - start_time
        
        # Test Cosine KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_cosine(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        cosine_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_cosine(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        cosine_cpu_time = time.time() - start_time
        
        # Test Dot Product KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_dot(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        dot_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_dot(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        dot_cpu_time = time.time() - start_time
        
        # Test Manhattan KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_manhattan(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        manhattan_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_manhattan(N, FIXED_DIMENSION, A_np, KMEANS_CLUSTERS)
        manhattan_cpu_time = time.time() - start_time
        
        # Calculate speedups
        l2_speedup = l2_cpu_time / l2_gpu_time
        cosine_speedup = cosine_cpu_time / cosine_gpu_time
        dot_speedup = dot_cpu_time / dot_gpu_time
        manhattan_speedup = manhattan_cpu_time / manhattan_gpu_time
        
        # Store results
        results['vector_count'].append(N)
        results['l2_gpu_time'].append(l2_gpu_time)
        results['cosine_gpu_time'].append(cosine_gpu_time)
        results['dot_gpu_time'].append(dot_gpu_time)
        results['manhattan_gpu_time'].append(manhattan_gpu_time)
        results['l2_cpu_time'].append(l2_cpu_time)
        results['cosine_cpu_time'].append(cosine_cpu_time)
        results['dot_cpu_time'].append(dot_cpu_time)
        results['manhattan_cpu_time'].append(manhattan_cpu_time)
        results['l2_speedup'].append(l2_speedup)
        results['cosine_speedup'].append(cosine_speedup)
        results['dot_speedup'].append(dot_speedup)
        results['manhattan_speedup'].append(manhattan_speedup)
        
        # Print results
        print(f"    L2 KMeans - GPU: {l2_gpu_time:.6f}s, CPU: {l2_cpu_time:.6f}s, Speedup: {l2_speedup:.2f}x")
        print(f"    Cosine KMeans - GPU: {cosine_gpu_time:.6f}s, CPU: {cosine_cpu_time:.6f}s, Speedup: {cosine_speedup:.2f}x")
        print(f"    Dot Product KMeans - GPU: {dot_gpu_time:.6f}s, CPU: {dot_cpu_time:.6f}s, Speedup: {dot_speedup:.2f}x")
        print(f"    Manhattan KMeans - GPU: {manhattan_gpu_time:.6f}s, CPU: {manhattan_cpu_time:.6f}s, Speedup: {manhattan_speedup:.2f}x")
    
    # Save and plot results
    pd.DataFrame(results).to_csv('results/kmeans_vector_count.csv', index=False)
    plot_and_save(results, 'KMeans Vector Count', 'kmeans_vector_count', 'vector_count')
    
    print("KMeans vector count test completed.")

def test_kmeans_dimension():
    """Test KMeans performance with varying vector dimensions."""
    print("Testing KMeans with varying vector dimensions...")
    
    # Results storage
    results = {
        'dimension': [],
        'l2_gpu_time': [], 'cosine_gpu_time': [], 'dot_gpu_time': [], 'manhattan_gpu_time': [],
        'l2_cpu_time': [], 'cosine_cpu_time': [], 'dot_cpu_time': [], 'manhattan_cpu_time': [],
        'l2_speedup': [], 'cosine_speedup': [], 'dot_speedup': [], 'manhattan_speedup': []
    }
    
    for D in VECTOR_DIMENSIONS:
        print(f"  Testing with dimension {D}...")
        
        # Generate dataset
        A_np = generate_dataset(FIXED_VECTORS, D)
        
        # Test L2 KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_l2(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        l2_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_l2(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        l2_cpu_time = time.time() - start_time
        
        # Test Cosine KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_cosine(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        cosine_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_cosine(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        cosine_cpu_time = time.time() - start_time
        
        # Test Dot Product KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_dot(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        dot_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_dot(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        dot_cpu_time = time.time() - start_time
        
        # Test Manhattan KMeans
        torch.cuda.synchronize()
        start_time = time.time()
        our_kmeans_manhattan(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        torch.cuda.synchronize()
        manhattan_gpu_time = time.time() - start_time
        
        start_time = time.time()
        cpu_kmeans_manhattan(FIXED_VECTORS, D, A_np, KMEANS_CLUSTERS)
        manhattan_cpu_time = time.time() - start_time
        
        # Calculate speedups
        l2_speedup = l2_cpu_time / l2_gpu_time
        cosine_speedup = cosine_cpu_time / cosine_gpu_time
        dot_speedup = dot_cpu_time / dot_gpu_time
        manhattan_speedup = manhattan_cpu_time / manhattan_gpu_time
        
        # Store results
        results['dimension'].append(D)
        results['l2_gpu_time'].append(l2_gpu_time)
        results['cosine_gpu_time'].append(cosine_gpu_time)
        results['dot_gpu_time'].append(dot_gpu_time)
        results['manhattan_gpu_time'].append(manhattan_gpu_time)
        results['l2_cpu_time'].append(l2_cpu_time)
        results['cosine_cpu_time'].append(cosine_cpu_time)
        results['dot_cpu_time'].append(dot_cpu_time)
        results['manhattan_cpu_time'].append(manhattan_cpu_time)
        results['l2_speedup'].append(l2_speedup)
        results['cosine_speedup'].append(cosine_speedup)
        results['dot_speedup'].append(dot_speedup)
        results['manhattan_speedup'].append(manhattan_speedup)
        
        # Print results
        print(f"    L2 KMeans - GPU: {l2_gpu_time:.6f}s, CPU: {l2_cpu_time:.6f}s, Speedup: {l2_speedup:.2f}x")
        print(f"    Cosine KMeans - GPU: {cosine_gpu_time:.6f}s, CPU: {cosine_cpu_time:.6f}s, Speedup: {cosine_speedup:.2f}x")
        print(f"    Dot Product KMeans - GPU: {dot_gpu_time:.6f}s, CPU: {dot_cpu_time:.6f}s, Speedup: {dot_speedup:.2f}x")
        print(f"    Manhattan KMeans - GPU: {manhattan_gpu_time:.6f}s, CPU: {manhattan_cpu_time:.6f}s, Speedup: {manhattan_speedup:.2f}x")
    
    # Save and plot results
    pd.DataFrame(results).to_csv('results/kmeans_dimension.csv', index=False)
    plot_and_save(results, 'KMeans Dimension', 'kmeans_dimension', 'dimension', is_dimension=True)
    
    print("KMeans dimension test completed.")

# -----------------------------------------------------------------------------
# CPU Implementation of KNN
# -----------------------------------------------------------------------------
def cpu_knn_l2(N, D, A, X, K):
    """Compute K-nearest neighbors using L2 distance on CPU."""
    distances = cpu_l2_distance(A, X)
    indices = np.argsort(distances)[:K]
    return indices.tolist()

def cpu_knn_cosine(N, D, A, X, K):
    """Compute K-nearest neighbors using Cosine distance on CPU."""
    distances = cpu_cosine_distance(A, X)
    indices = np.argsort(distances)[:K]
    return indices.tolist()

def cpu_knn_dot(N, D, A, X, K):
    """Compute K-nearest neighbors using Dot Product distance on CPU."""
    distances = cpu_dot_distance(A, X)
    indices = np.argsort(distances)[:K]
    return indices.tolist()

def cpu_knn_manhattan(N, D, A, X, K):
    """Compute K-nearest neighbors using Manhattan distance on CPU."""
    distances = cpu_manhattan_distance(A, X)
    indices = np.argsort(distances)[:K]
    return indices.tolist()

# -----------------------------------------------------------------------------
# KNN Test Functions
# -----------------------------------------------------------------------------
def test_knn_dimension():
    """Test KNN performance with varying vector dimensions."""
    print("Testing KNN with varying vector dimensions...")
    
    # Fixed number of vectors
    N = 10000
    
    # Vector dimensions to test (powers of 2)
    dimensions = [2, 8, 32, 128, 512, 2048, 8192, 32768]
    
    # Fixed K value
    K = 10
    
    # Results storage
    results = {
        'dimension': [],
        'l2_gpu_time': [], 'cosine_gpu_time': [], 'dot_gpu_time': [], 'manhattan_gpu_time': [],
        'l2_cpu_time': [], 'cosine_cpu_time': [], 'dot_cpu_time': [], 'manhattan_cpu_time': [],
        'l2_speedup': [], 'cosine_speedup': [], 'dot_speedup': [], 'manhattan_speedup': []
    }
    
    for D in dimensions:
        print(f"  Testing with dimension {D}...")
        
        # Generate dataset
        A_np = generate_dataset(N, D)
        X_np = generate_dataset(1, D)[0]  # Single query vector
        
        # Test L2 KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_l2(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        l2_gpu_time = time.time() - start_time
        
        # Test L2 KNN on CPU
        start_time = time.time()
        cpu_knn_l2(N, D, A_np, X_np, K)
        l2_cpu_time = time.time() - start_time
        
        # Test Cosine KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_cosine(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        cosine_gpu_time = time.time() - start_time
        
        # Test Cosine KNN on CPU
        start_time = time.time()
        cpu_knn_cosine(N, D, A_np, X_np, K)
        cosine_cpu_time = time.time() - start_time
        
        # Test Dot Product KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_dot(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        dot_gpu_time = time.time() - start_time
        
        # Test Dot Product KNN on CPU
        start_time = time.time()
        cpu_knn_dot(N, D, A_np, X_np, K)
        dot_cpu_time = time.time() - start_time
        
        # Test Manhattan KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_manhattan(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        manhattan_gpu_time = time.time() - start_time
        
        # Test Manhattan KNN on CPU
        start_time = time.time()
        cpu_knn_manhattan(N, D, A_np, X_np, K)
        manhattan_cpu_time = time.time() - start_time
        
        # Calculate speedups
        l2_speedup = l2_cpu_time / l2_gpu_time
        cosine_speedup = cosine_cpu_time / cosine_gpu_time
        dot_speedup = dot_cpu_time / dot_gpu_time
        manhattan_speedup = manhattan_cpu_time / manhattan_gpu_time
        
        # Store results
        results['dimension'].append(D)
        results['l2_gpu_time'].append(l2_gpu_time)
        results['cosine_gpu_time'].append(cosine_gpu_time)
        results['dot_gpu_time'].append(dot_gpu_time)
        results['manhattan_gpu_time'].append(manhattan_gpu_time)
        results['l2_cpu_time'].append(l2_cpu_time)
        results['cosine_cpu_time'].append(cosine_cpu_time)
        results['dot_cpu_time'].append(dot_cpu_time)
        results['manhattan_cpu_time'].append(manhattan_cpu_time)
        results['l2_speedup'].append(l2_speedup)
        results['cosine_speedup'].append(cosine_speedup)
        results['dot_speedup'].append(dot_speedup)
        results['manhattan_speedup'].append(manhattan_speedup)
        
        # Print results
        print(f"    L2 KNN - GPU: {l2_gpu_time:.6f}s, CPU: {l2_cpu_time:.6f}s, Speedup: {l2_speedup:.2f}x")
        print(f"    Cosine KNN - GPU: {cosine_gpu_time:.6f}s, CPU: {cosine_cpu_time:.6f}s, Speedup: {cosine_speedup:.2f}x")
        print(f"    Dot Product KNN - GPU: {dot_gpu_time:.6f}s, CPU: {dot_cpu_time:.6f}s, Speedup: {dot_speedup:.2f}x")
        print(f"    Manhattan KNN - GPU: {manhattan_gpu_time:.6f}s, CPU: {manhattan_cpu_time:.6f}s, Speedup: {manhattan_speedup:.2f}x")
    
    # Save and plot results
    pd.DataFrame(results).to_csv('results/knn_dimension.csv', index=False)
    plot_and_save(results, 'KNN Dimension', 'knn_dimension', 'dimension', is_dimension=True)
    
    print("KNN dimension test completed.")

def test_knn_vector_count():
    """Test KNN performance with varying vector counts."""
    print("Testing KNN with varying vector counts...")
    
    # Vector counts to test
    vector_counts = [4000, 10000, 50000, 100000, 500000, 1000000, 4000000]
    
    # Fixed dimension for the test
    D = 128
    
    # Fixed K value
    K = 10
    
    # Results storage
    results = {
        'vector_count': [],
        'l2_gpu_time': [], 'cosine_gpu_time': [], 'dot_gpu_time': [], 'manhattan_gpu_time': [],
        'l2_cpu_time': [], 'cosine_cpu_time': [], 'dot_cpu_time': [], 'manhattan_cpu_time': [],
        'l2_speedup': [], 'cosine_speedup': [], 'dot_speedup': [], 'manhattan_speedup': []
    }
    
    for N in vector_counts:
        print(f"  Testing with {N} vectors...")
        
        # Generate dataset
        A_np = generate_dataset(N, D)
        X_np = generate_dataset(1, D)[0]  # Single query vector
        
        # Test L2 KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_l2(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        l2_gpu_time = time.time() - start_time
        
        # Test L2 KNN on CPU
        start_time = time.time()
        cpu_knn_l2(N, D, A_np, X_np, K)
        l2_cpu_time = time.time() - start_time
        
        # Test Cosine KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_cosine(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        cosine_gpu_time = time.time() - start_time
        
        # Test Cosine KNN on CPU
        start_time = time.time()
        cpu_knn_cosine(N, D, A_np, X_np, K)
        cosine_cpu_time = time.time() - start_time
        
        # Test Dot Product KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_dot(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        dot_gpu_time = time.time() - start_time
        
        # Test Dot Product KNN on CPU
        start_time = time.time()
        cpu_knn_dot(N, D, A_np, X_np, K)
        dot_cpu_time = time.time() - start_time
        
        # Test Manhattan KNN on GPU
        torch.cuda.synchronize()
        start_time = time.time()
        our_knn_manhattan(N, D, A_np, X_np, K)
        torch.cuda.synchronize()
        manhattan_gpu_time = time.time() - start_time
        
        # Test Manhattan KNN on CPU
        start_time = time.time()
        cpu_knn_manhattan(N, D, A_np, X_np, K)
        manhattan_cpu_time = time.time() - start_time
        
        # Calculate speedups
        l2_speedup = l2_cpu_time / l2_gpu_time
        cosine_speedup = cosine_cpu_time / cosine_gpu_time
        dot_speedup = dot_cpu_time / dot_gpu_time
        manhattan_speedup = manhattan_cpu_time / manhattan_gpu_time
        
        # Store results
        results['vector_count'].append(N)
        results['l2_gpu_time'].append(l2_gpu_time)
        results['cosine_gpu_time'].append(cosine_gpu_time)
        results['dot_gpu_time'].append(dot_gpu_time)
        results['manhattan_gpu_time'].append(manhattan_gpu_time)
        results['l2_cpu_time'].append(l2_cpu_time)
        results['cosine_cpu_time'].append(cosine_cpu_time)
        results['dot_cpu_time'].append(dot_cpu_time)
        results['manhattan_cpu_time'].append(manhattan_cpu_time)
        results['l2_speedup'].append(l2_speedup)
        results['cosine_speedup'].append(cosine_speedup)
        results['dot_speedup'].append(dot_speedup)
        results['manhattan_speedup'].append(manhattan_speedup)
        
        # Print results
        print(f"    L2 KNN - GPU: {l2_gpu_time:.6f}s, CPU: {l2_cpu_time:.6f}s, Speedup: {l2_speedup:.2f}x")
        print(f"    Cosine KNN - GPU: {cosine_gpu_time:.6f}s, CPU: {cosine_cpu_time:.6f}s, Speedup: {cosine_speedup:.2f}x")
        print(f"    Dot Product KNN - GPU: {dot_gpu_time:.6f}s, CPU: {dot_cpu_time:.6f}s, Speedup: {dot_speedup:.2f}x")
        print(f"    Manhattan KNN - GPU: {manhattan_gpu_time:.6f}s, CPU: {manhattan_cpu_time:.6f}s, Speedup: {manhattan_speedup:.2f}x")
    
    # Save and plot results
    pd.DataFrame(results).to_csv('results/knn_vector_count.csv', index=False)
    plot_and_save(results, 'KNN Vector Count', 'knn_vector_count', 'vector_count')
    
    print("KNN vector count test completed.")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """Run all experiments and report results."""
    print("Starting GPU Vector Search Experiments...")
    print("=" * 80)
    
    
    # Make sure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    
    # Record system info
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton available: {hasattr(torch, 'cuda') and torch.cuda.is_available()}")
    print("=" * 80)
    
    # Run experiments
    gpu_prewarm()
    print("\n1. Distance Functions Scaling Test")
    print("-" * 80)
    test_distance_functions_scaling()
    
    gpu_prewarm()
    print("\n2. KMeans Vector Count Test")
    print("-" * 80)
    test_kmeans_vector_count()
    
    gpu_prewarm()
    print("\n3. KMeans Dimension Test")
    print("-" * 80)
    test_kmeans_dimension()
    
    gpu_prewarm()
    print("\n4. KNN Vector Count Test")
    print("-" * 80)
    test_knn_vector_count()
    
    gpu_prewarm()
    print("\n5. KNN Dimension Test")
    print("-" * 80)
    test_knn_dimension()
    
    print("\nAll experiments completed successfully.")
    print("Results are saved in the 'results/' directory, and figures in the 'figures/' directory.")
    
    # Generate summary report

def generate_summary_report():
    """Generate a comprehensive summary report of all experiments."""
    print("\nGenerating summary report...")
    
    # Create a summary dataframe for each experiment type
    summary = {
        'experiment': [],
        'max_l2_speedup': [], 'max_cosine_speedup': [], 'max_dot_speedup': [], 'max_manhattan_speedup': [],
        'avg_l2_speedup': [], 'avg_cosine_speedup': [], 'avg_dot_speedup': [], 'avg_manhattan_speedup': [],
        'best_dimension': [], 'best_vector_count': []
    }
    
    # Process distance function results
    if os.path.exists('results/distance_functions_scaling.csv'):
        df = pd.read_csv('results/distance_functions_scaling.csv')
        summary['experiment'].append('Distance Functions')
        summary['max_l2_speedup'].append(df['l2_speedup'].max())
        summary['max_cosine_speedup'].append(df['cosine_speedup'].max())
        summary['max_dot_speedup'].append(df['dot_speedup'].max())
        summary['max_manhattan_speedup'].append(df['manhattan_speedup'].max())
        summary['avg_l2_speedup'].append(df['l2_speedup'].mean())
        summary['avg_cosine_speedup'].append(df['cosine_speedup'].mean())
        summary['avg_dot_speedup'].append(df['dot_speedup'].mean())
        summary['avg_manhattan_speedup'].append(df['manhattan_speedup'].mean())
        summary['best_dimension'].append('N/A')
        summary['best_vector_count'].append(df.loc[df['l2_speedup'].idxmax()]['vector_count'])
    
    # Process KMeans vector count results
    if os.path.exists('results/kmeans_vector_count.csv'):
        df = pd.read_csv('results/kmeans_vector_count.csv')
        summary['experiment'].append('KMeans Vector Count')
        summary['max_l2_speedup'].append(df['l2_speedup'].max())
        summary['max_cosine_speedup'].append(df['cosine_speedup'].max())
        summary['max_dot_speedup'].append(df['dot_speedup'].max())
        summary['max_manhattan_speedup'].append(df['manhattan_speedup'].max())
        summary['avg_l2_speedup'].append(df['l2_speedup'].mean())
        summary['avg_cosine_speedup'].append(df['cosine_speedup'].mean())
        summary['avg_dot_speedup'].append(df['dot_speedup'].mean())
        summary['avg_manhattan_speedup'].append(df['manhattan_speedup'].mean())
        summary['best_dimension'].append('N/A')
        summary['best_vector_count'].append(df.loc[df['l2_speedup'].idxmax()]['vector_count'])
    
    # Process KMeans dimension results
    if os.path.exists('results/kmeans_dimension.csv'):
        df = pd.read_csv('results/kmeans_dimension.csv')
        summary['experiment'].append('KMeans Dimension')
        summary['max_l2_speedup'].append(df['l2_speedup'].max())
        summary['max_cosine_speedup'].append(df['cosine_speedup'].max())
        summary['max_dot_speedup'].append(df['dot_speedup'].max())
        summary['max_manhattan_speedup'].append(df['manhattan_speedup'].max())
        summary['avg_l2_speedup'].append(df['l2_speedup'].mean())
        summary['avg_cosine_speedup'].append(df['cosine_speedup'].mean())
        summary['avg_dot_speedup'].append(df['dot_speedup'].mean())
        summary['avg_manhattan_speedup'].append(df['manhattan_speedup'].mean())
        summary['best_dimension'].append(df.loc[df['l2_speedup'].idxmax()]['dimension'])
        summary['best_vector_count'].append('N/A')
    
    # Process KNN vector count results
    if os.path.exists('results/knn_vector_count.csv'):
        df = pd.read_csv('results/knn_vector_count.csv')
        summary['experiment'].append('KNN Vector Count')
        summary['max_l2_speedup'].append(df['l2_speedup'].max())
        summary['max_cosine_speedup'].append(df['cosine_speedup'].max())
        summary['max_dot_speedup'].append(df['dot_speedup'].max())
        summary['max_manhattan_speedup'].append(df['manhattan_speedup'].max())
        summary['avg_l2_speedup'].append(df['l2_speedup'].mean())
        summary['avg_cosine_speedup'].append(df['cosine_speedup'].mean())
        summary['avg_dot_speedup'].append(df['dot_speedup'].mean())
        summary['avg_manhattan_speedup'].append(df['manhattan_speedup'].mean())
        summary['best_dimension'].append('N/A')
        summary['best_vector_count'].append(df.loc[df['l2_speedup'].idxmax()]['vector_count'])
    
    # Process KNN dimension results
    if os.path.exists('results/knn_dimension.csv'):
        df = pd.read_csv('results/knn_dimension.csv')
        summary['experiment'].append('KNN Dimension')
        summary['max_l2_speedup'].append(df['l2_speedup'].max())
        summary['max_cosine_speedup'].append(df['cosine_speedup'].max())
        summary['max_dot_speedup'].append(df['dot_speedup'].max())
        summary['max_manhattan_speedup'].append(df['manhattan_speedup'].max())
        summary['avg_l2_speedup'].append(df['l2_speedup'].mean())
        summary['avg_cosine_speedup'].append(df['cosine_speedup'].mean())
        summary['avg_dot_speedup'].append(df['dot_speedup'].mean())
        summary['avg_manhattan_speedup'].append(df['manhattan_speedup'].mean())
        summary['best_dimension'].append(df.loc[df['l2_speedup'].idxmax()]['dimension'])
        summary['best_vector_count'].append('N/A')
    
    # Create summary dataframe and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/summary.csv', index=False)
    
    # Create summary visualization
    plt.figure(figsize=(14, 10))
    
    # Plot max speedups by experiment
    plt.subplot(2, 1, 1)
    experiments = summary_df['experiment']
    x = np.arange(len(experiments))
    width = 0.2
    
    plt.bar(x - width*1.5, summary_df['max_l2_speedup'], width, label='L2')
    plt.bar(x - width*0.5, summary_df['max_cosine_speedup'], width, label='Cosine')
    plt.bar(x + width*0.5, summary_df['max_dot_speedup'], width, label='Dot Product')
    plt.bar(x + width*1.5, summary_df['max_manhattan_speedup'], width, label='Manhattan')
    
    plt.ylabel('Maximum Speedup (CPU/GPU)')
    plt.title('Maximum GPU Speedup by Experiment Type and Distance Metric')
    plt.xticks(x, experiments, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average speedups by experiment
    plt.subplot(2, 1, 2)
    plt.bar(x - width*1.5, summary_df['avg_l2_speedup'], width, label='L2')
    plt.bar(x - width*0.5, summary_df['avg_cosine_speedup'], width, label='Cosine')
    plt.bar(x + width*0.5, summary_df['avg_dot_speedup'], width, label='Dot Product')
    plt.bar(x + width*1.5, summary_df['avg_manhattan_speedup'], width, label='Manhattan')
    
    plt.ylabel('Average Speedup (CPU/GPU)')
    plt.title('Average GPU Speedup by Experiment Type and Distance Metric')
    plt.xticks(x, experiments, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('figures/summary.png')
    
    print(f"Summary report saved to 'results/summary.csv' and 'figures/summary.png'")
    
    # Print key findings
    print("\nKey Findings:")
    best_overall = summary_df.loc[summary_df['max_l2_speedup'].idxmax()]
    print(f"- Best overall GPU speedup: {best_overall['max_l2_speedup']:.2f}x for L2 distance in {best_overall['experiment']}")
    
    for metric in ['l2', 'cosine', 'dot', 'manhattan']:
        best_experiment = summary_df.loc[summary_df[f'max_{metric}_speedup'].idxmax()]
        print(f"- Best {metric.capitalize()} speedup: {best_experiment[f'max_{metric}_speedup']:.2f}x in {best_experiment['experiment']}")

# -----------------------------------------------------------------------------
# GPU Pre-warming Function
# -----------------------------------------------------------------------------
def gpu_prewarm():
    """Pre-warm the GPU by executing a dummy matrix multiplication."""
    if torch.cuda.is_available():
        print("Pre-warming GPU...")
        dummy_a = torch.randn(1024, 1024, device="cuda")
        dummy_b = torch.randn(1024, 1024, device="cuda")
        _ = torch.mm(dummy_a, dummy_b)
        torch.cuda.synchronize()
        print("GPU warming complete.")

if __name__ == "__main__":
    main()