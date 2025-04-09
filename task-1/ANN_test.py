#!/usr/bin/env python3
"""
ANN_test.py

This script tests our ANN (Approximate Nearest Neighbor) implementations by scaling
the vector dimension from 2 to 32768. For each metric (L2, Cosine, Dot, and Manhattan)
the script measures:
  - The offline KMeans clustering time (excluded from query timing),
  - The GPU query time for our ANN search,
  - The CPU query time for our ANN search,
  - The recall of each ANN result compared to the GPU exact KNN result (ground truth), and
  - The GPU–CPU speedup factor.

Offline clustering divides the vector space into a large number of clusters (n_clusters)
while only probing a subset (n_probe) during query time. For both cosine and dot metrics,
the dataset (and query vectors) are normalized to help improve recall.
"""

import numpy as np
import torch
import time

# Import functions from task.py.
from task import (
    our_ann_l2, our_ann_cosine, our_ann_dot, our_ann_manhattan,
    our_knn_l2, our_knn_cosine, our_knn_dot, our_knn_manhattan,
    our_kmeans_l2, our_kmeans_cosine, our_kmeans_dot, our_kmeans_manhattan,
    l2_distance_kernel, cosine_distance_kernel, dot_distance_kernel, manhattan_distance_kernel,
    gpu_topk
)

# Import CPU KNN functions from experiments.py.
from experiments import cpu_knn_l2, cpu_knn_cosine, cpu_knn_dot, cpu_knn_manhattan

def compute_params(N, D, K, metric):
    """
    Improved clustering parameter selection without using built-in functions
    """
    # Calculate reasonable number of clusters based on data size
    if N <= 500:
        # For very small datasets, use fewer clusters
        n_clusters = max(int(np.sqrt(N) * 0.5), 3)
    elif N <= 2000:
        # For small datasets
        n_clusters = max(int(np.sqrt(N) * 0.7), 8)
    else:
        # For larger datasets
        n_clusters = max(int(np.sqrt(N) * 0.9), 15)
    
    # Expected average points per cluster
    expected_points_per_cluster = N / n_clusters
    
    # Calculate K2 based on expected cluster size
    # K2 should never exceed the expected points per cluster
    base_K2 = min(int(expected_points_per_cluster * 0.7), 100)
    
    # Metric-specific adjustments
    if metric == 'cosine':
        # For cosine, we need more clusters but smaller K2
        n_clusters = int(n_clusters * 1.3)
        expected_points_per_cluster = N / n_clusters
        K2 = min(max(base_K2, K * 4), int(expected_points_per_cluster * 0.8))
        # Increase n_probe for cosine to improve recall
        n_probe = min(max(int(n_clusters * 0.4), 4), n_clusters)
    elif metric == 'dot':
        # For dot product, similar to cosine
        n_clusters = int(n_clusters * 1.2)
        expected_points_per_cluster = N / n_clusters
        K2 = min(max(base_K2, K * 3), int(expected_points_per_cluster * 0.8))
        n_probe = min(max(int(n_clusters * 0.35), 3), n_clusters)
    else:  # L2 or Manhattan
        K2 = min(max(base_K2, K * 2), int(expected_points_per_cluster * 0.7))
        n_probe = min(max(int(n_clusters * 0.3), 3), n_clusters)
    
    # Dimension-specific adjustments
    if D >= 512:
        # For high dimensions, increase n_probe
        n_probe = min(max(n_probe, int(n_clusters * 0.5)), n_clusters)
        
    # Ensure K2 is at least K to avoid missing neighbors
    K2 = max(K2, K)
    
    # Safeguard: ensure n_probe doesn't exceed n_clusters
    n_probe = min(n_probe, n_clusters)
    
    # Log parameters for diagnostic purposes
    print(f"Parameters for {metric} metric with N={N}, D={D}:")
    print(f"  n_clusters={n_clusters}, expected points per cluster={expected_points_per_cluster:.1f}")
    print(f"  n_probe={n_probe}, K2={K2}")
    
    return n_clusters, n_probe, K2

def normalize_vectors(A_np, X_np=None, metric='l2'):
    """
    Consistently normalize vectors for cosine and dot metrics
    """
    if metric not in ('cosine', 'dot'):
        if X_np is not None:
            return A_np, X_np
        return A_np
    
    # Normalize dataset
    norms = np.linalg.norm(A_np, axis=1, keepdims=True)
    # Handle zero vectors
    mask = norms < 1e-8
    norms[mask] = 1.0
    A_np_norm = A_np / norms
    
    # Normalize query if provided
    if X_np is not None:
        X_norm = np.linalg.norm(X_np)
        if X_norm < 1e-8:
            X_norm = 1.0
        X_np_norm = X_np / X_norm
        return A_np_norm, X_np_norm
    
    return A_np_norm

def offline_kmeans(metric, A_np, n_clusters, D, N):
    """
    Optimized offline clustering function that keeps results on GPU.
    
    Parameters:
        metric: Distance metric to use ('l2', 'cosine', 'dot', 'manhattan')
        A_np: Dataset as numpy array [N, D]
        n_clusters: Number of clusters to create
        D: Vector dimension
        N: Number of vectors
        
    Returns:
        Tuple of (cluster_ids, centroids) where both are PyTorch tensors on GPU
    """
    # Call the appropriate kmeans function
    if metric == 'l2':
        cpu_cluster_ids, cpu_centroids = our_kmeans_l2(N, D, A_np, n_clusters)
    elif metric == 'cosine':
        cpu_cluster_ids, cpu_centroids = our_kmeans_cosine(N, D, A_np, n_clusters)
    elif metric == 'dot':
        cpu_cluster_ids, cpu_centroids = our_kmeans_dot(N, D, A_np, n_clusters)
    elif metric == 'manhattan':
        cpu_cluster_ids, cpu_centroids = our_kmeans_manhattan(N, D, A_np, n_clusters)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    # Convert results to GPU tensors directly
    # This prevents redundant CPU->GPU transfers later
    cluster_ids_gpu = torch.tensor(cpu_cluster_ids, device="cuda", dtype=torch.int64)
    centroids_gpu = torch.tensor(cpu_centroids, device="cuda", dtype=torch.float32)
    
    return cluster_ids_gpu, centroids_gpu

def ann_query(metric, A_np, X_np, K, n_probe, K2, cluster_ids, centroids, D):
    """
    Optimized ANN query function that minimizes CPU-GPU transfers and keeps data on the GPU,
    maintaining the same method name as the original for drop-in replacement.
    
    Parameters:
        metric: Distance metric ('l2', 'cosine', 'dot', 'manhattan')
        A_np: Dataset as numpy array [N, D]
        X_np: Query vector as numpy array [D]
        K: Number of nearest neighbors to find
        n_probe: Number of clusters to probe
        K2: Maximum candidates per cluster
        cluster_ids: Cluster assignments from kmeans
        centroids: Cluster centroids
        D: Vector dimension
        
    Returns:
        List of indices of the K nearest neighbors
    """
    # Convert all inputs to GPU tensors once
    if isinstance(A_np, np.ndarray):
        A = torch.tensor(A_np, device="cuda", dtype=torch.float32)
    else:
        A = A_np.cuda() if A_np.device.type != "cuda" else A_np
        
    if isinstance(X_np, np.ndarray):
        X = torch.tensor(X_np, device="cuda", dtype=torch.float32)
    else:
        X = X_np.cuda() if X_np.device.type != "cuda" else X_np
        
    if isinstance(cluster_ids, list) or isinstance(cluster_ids, np.ndarray):
        cluster_ids_tensor = torch.tensor(cluster_ids, device="cuda", dtype=torch.int64)
    else:
        cluster_ids_tensor = cluster_ids.cuda() if cluster_ids.device.type != "cuda" else cluster_ids
        
    if isinstance(centroids, np.ndarray):
        centroids_tensor = torch.tensor(centroids, device="cuda", dtype=torch.float32)
    else:
        centroids_tensor = centroids.cuda() if centroids.device.type != "cuda" else centroids
    
    # Normalize vectors for cosine/dot metrics (once)
    if metric in ('cosine', 'dot'):
        # Normalize A efficiently using existing operations
        A_norms = torch.norm(A, dim=1, keepdim=True)
        A_norms = torch.clamp(A_norms, min=1e-8)  # Avoid division by zero
        A = A / A_norms
        
        # Normalize X
        X_norm = torch.norm(X)
        X_norm = max(X_norm, 1e-8)  # Avoid division by zero
        X = X / X_norm
        
        # Normalize centroids
        centroids_norms = torch.norm(centroids_tensor, dim=1, keepdim=True)
        centroids_norms = torch.clamp(centroids_norms, min=1e-8)
        centroids_tensor = centroids_tensor / centroids_norms
    
    # Compute distances to centroids using the appropriate kernel
    n_centroids = centroids_tensor.shape[0]
    centroid_distances = torch.empty(n_centroids, device="cuda", dtype=torch.float32)
    grid = (n_centroids,)
    block_size = 256 # if D > 512 else 128
    
    if metric == 'l2':
        l2_distance_kernel[grid](centroids_tensor, X, centroid_distances, D, centroids_tensor.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'cosine':
        cosine_distance_kernel[grid](centroids_tensor, X, centroid_distances, D, centroids_tensor.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'dot':
        dot_distance_kernel[grid](centroids_tensor, X, centroid_distances, D, centroids_tensor.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'manhattan':
        manhattan_distance_kernel[grid](centroids_tensor, X, centroid_distances, D, centroids_tensor.stride(0), BLOCK_SIZE=block_size)
    
    # Adjust n_probe for cosine/dot metrics to improve recall
    actual_n_probe = n_probe
    if metric in ('cosine', 'dot'):
        distances_mean = torch.mean(centroid_distances)
        distances_std = torch.std(centroid_distances)
        
        if distances_std < distances_mean * 0.5:
            actual_n_probe = min(n_probe * 2, n_centroids)
        else:
            actual_n_probe = min(n_probe * 1.5, n_centroids)
    
    # Find nearest clusters using gpu_topk
    nearest_clusters = gpu_topk(centroid_distances, min(int(actual_n_probe), n_centroids))
    
    # Collect candidates from selected clusters
    all_candidate_indices = []
    
    for cluster_idx in nearest_clusters:
        idx = cluster_idx.item()
        # Find points belonging to this cluster
        cluster_point_indices = (cluster_ids_tensor == idx).nonzero(as_tuple=True)[0]
        
        if cluster_point_indices.size(0) > 0:
            # Extract points for this cluster
            cluster_points = A.index_select(0, cluster_point_indices)
            
            # Compute distances within cluster
            distances = torch.empty(cluster_point_indices.size(0), device="cuda", dtype=torch.float32)
            cluster_grid = (cluster_point_indices.size(0),)
            
            if metric == 'l2':
                l2_distance_kernel[cluster_grid](cluster_points, X, distances, D, cluster_points.stride(0), BLOCK_SIZE=block_size)
            elif metric == 'cosine':
                cosine_distance_kernel[cluster_grid](cluster_points, X, distances, D, cluster_points.stride(0), BLOCK_SIZE=block_size)
            elif metric == 'dot':
                dot_distance_kernel[cluster_grid](cluster_points, X, distances, D, cluster_points.stride(0), BLOCK_SIZE=block_size)
            elif metric == 'manhattan':
                manhattan_distance_kernel[cluster_grid](cluster_points, X, distances, D, cluster_points.stride(0), BLOCK_SIZE=block_size)
            
            # Select top-K2 within cluster
            actual_K2 = min(K2, cluster_point_indices.size(0))
            
            # For cosine/dot with small clusters, take more points
            if metric in ('cosine', 'dot') and cluster_point_indices.size(0) < K * 3:
                actual_K2 = cluster_point_indices.size(0)
            
            if actual_K2 > 0:
                topk_indices = gpu_topk(distances, actual_K2)
                selected_indices = cluster_point_indices.index_select(0, topk_indices)
                all_candidate_indices.append(selected_indices)
    
    # Fall back to exact KNN if no candidates found
    if not all_candidate_indices:
        all_distances = torch.empty(A.shape[0], device="cuda", dtype=torch.float32)
        grid = (A.shape[0],)
        
        if metric == 'l2':
            l2_distance_kernel[grid](A, X, all_distances, D, A.stride(0), BLOCK_SIZE=block_size)
        elif metric == 'cosine':
            cosine_distance_kernel[grid](A, X, all_distances, D, A.stride(0), BLOCK_SIZE=block_size)
        elif metric == 'dot':
            dot_distance_kernel[grid](A, X, all_distances, D, A.stride(0), BLOCK_SIZE=block_size)
        elif metric == 'manhattan':
            manhattan_distance_kernel[grid](A, X, all_distances, D, A.stride(0), BLOCK_SIZE=block_size)
            
        topk_indices = gpu_topk(all_distances, min(K, A.shape[0]))
        return topk_indices.cpu().tolist()
    
    # Efficiently merge candidate indices
    merged_candidates = torch.cat(all_candidate_indices)
    
    # Manage memory for large candidate sets
    if merged_candidates.size(0) > 80000:
        perm = torch.randperm(merged_candidates.size(0), device="cuda")[:80000]
        merged_candidates = merged_candidates[perm]
    
    # Extract candidate vectors
    candidate_points = A.index_select(0, merged_candidates)
    
    # Compute final distances
    candidate_distances = torch.empty(candidate_points.shape[0], device="cuda", dtype=torch.float32)
    candidate_grid = (candidate_points.shape[0],)
    
    if metric == 'l2':
        l2_distance_kernel[candidate_grid](candidate_points, X, candidate_distances, D, candidate_points.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'cosine':
        cosine_distance_kernel[candidate_grid](candidate_points, X, candidate_distances, D, candidate_points.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'dot':
        dot_distance_kernel[candidate_grid](candidate_points, X, candidate_distances, D, candidate_points.stride(0), BLOCK_SIZE=block_size)
    elif metric == 'manhattan':
        manhattan_distance_kernel[candidate_grid](candidate_points, X, candidate_distances, D, candidate_points.stride(0), BLOCK_SIZE=block_size)
    
    # Get final top-K
    final_topk_indices = gpu_topk(candidate_distances, min(K, candidate_distances.size(0)))
    final_indices = merged_candidates.index_select(0, final_topk_indices)
    
    # Only transfer the final result back to CPU
    return final_indices.cpu().tolist()


def cpu_ann_query(metric, A_np, X_np, K, n_probe, K2, cluster_ids, centroids, D):
    """
    Adjusted CPU ANN implementation compatible with the GPU-optimized version.
    This function handles both GPU tensors and numpy arrays as inputs.
    
    Parameters:
        metric: Distance metric ('l2', 'cosine', 'dot', 'manhattan')
        A_np: Dataset as numpy array [N, D]
        X_np: Query vector as numpy array [D]
        K: Number of nearest neighbors to find
        n_probe: Number of clusters to probe
        K2: Maximum candidates per cluster
        cluster_ids: Cluster assignments from kmeans (can be tensor or numpy array)
        centroids: Cluster centroids (can be tensor or numpy array)
        D: Vector dimension
        
    Returns:
        List of indices of the K nearest neighbors
    """
    import numpy as np
    import torch
    
    # Convert tensors to numpy if needed
    if isinstance(A_np, torch.Tensor):
        A_np = A_np.cpu().numpy()
        
    if isinstance(X_np, torch.Tensor):
        X_np = X_np.cpu().numpy()
        
    if isinstance(cluster_ids, torch.Tensor):
        cluster_ids = cluster_ids.cpu().numpy()
    elif isinstance(cluster_ids, list):
        cluster_ids = np.array(cluster_ids)
        
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.cpu().numpy()
    
    # Normalize vectors for cosine/dot metrics (once)
    if metric in ('cosine', 'dot'):
        # Normalize A
        A_norms = np.linalg.norm(A_np, axis=1, keepdims=True)
        A_norms[A_norms < 1e-8] = 1.0  # Avoid division by zero
        A_norm = A_np / A_norms
        
        # Normalize X
        X_norm = np.linalg.norm(X_np)
        if X_norm < 1e-8:
            X_norm = 1.0
        X_norm = X_np / X_norm
        
        # Normalize centroids
        centroids_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_norms[centroids_norms < 1e-8] = 1.0
        centroids_norm = centroids / centroids_norms
    else:
        A_norm = A_np
        X_norm = X_np
        centroids_norm = centroids
    
    # Compute distances to centroids
    centroid_dists = np.zeros(centroids_norm.shape[0])
    
    if metric == 'l2':
        for i in range(centroids_norm.shape[0]):
            diff = centroids_norm[i] - X_norm
            centroid_dists[i] = np.sqrt(np.sum(diff * diff))
    elif metric == 'cosine':
        for i in range(centroids_norm.shape[0]):
            dot_product = np.sum(centroids_norm[i] * X_norm)
            centroid_dists[i] = 1.0 - dot_product
    elif metric == 'dot':
        for i in range(centroids_norm.shape[0]):
            dot_product = np.sum(centroids_norm[i] * X_norm)
            centroid_dists[i] = -dot_product
    elif metric == 'manhattan':
        for i in range(centroids_norm.shape[0]):
            centroid_dists[i] = np.sum(np.abs(centroids_norm[i] - X_norm))
    
    # Adjust n_probe for cosine/dot metrics
    actual_n_probe = n_probe
    if metric in ('cosine', 'dot'):
        distances_mean = np.mean(centroid_dists)
        distances_std = np.std(centroid_dists)
        
        if distances_std < distances_mean * 0.5:
            actual_n_probe = min(n_probe * 2, centroids_norm.shape[0])
        else:
            actual_n_probe = min(n_probe * 1.5, centroids_norm.shape[0])
    
    # Ensure actual_n_probe is an integer
    actual_n_probe = int(actual_n_probe)
    
    # Find nearest clusters
    nearest_cluster_indices = np.argsort(centroid_dists)[:actual_n_probe]
    
    # Collect candidates from selected clusters
    all_candidates = []
    
    for idx in nearest_cluster_indices:
        # Find points in this cluster
        cluster_points_indices = np.where(cluster_ids == idx)[0]
        
        if len(cluster_points_indices) > 0:
            # Get points for this cluster
            cluster_points = A_norm[cluster_points_indices]
            
            # Compute distances for points in this cluster
            dists = np.zeros(cluster_points.shape[0])
            
            if metric == 'l2':
                for i in range(cluster_points.shape[0]):
                    diff = cluster_points[i] - X_norm
                    dists[i] = np.sqrt(np.sum(diff * diff))
            elif metric == 'cosine':
                for i in range(cluster_points.shape[0]):
                    dot_product = np.sum(cluster_points[i] * X_norm)
                    dists[i] = 1.0 - dot_product
            elif metric == 'dot':
                for i in range(cluster_points.shape[0]):
                    dot_product = np.sum(cluster_points[i] * X_norm)
                    dists[i] = -dot_product
            elif metric == 'manhattan':
                for i in range(cluster_points.shape[0]):
                    dists[i] = np.sum(np.abs(cluster_points[i] - X_norm))
            
            # Adaptive K2 for this specific cluster
            actual_k2 = min(K2, len(cluster_points_indices))
            
            # For cosine/dot with small clusters, take all points
            if metric in ('cosine', 'dot') and len(cluster_points_indices) < K * 3:
                actual_k2 = len(cluster_points_indices)
            
            # Get top K2 points from this cluster
            if actual_k2 > 0:
                topk_indices = np.argsort(dists)[:actual_k2]
                selected_indices = cluster_points_indices[topk_indices]
                all_candidates.append(selected_indices)
    
    # Fall back to brute force if no candidates
    if not all_candidates:
        # Compute distances to all points
        all_dists = np.zeros(A_norm.shape[0])
        
        if metric == 'l2':
            for i in range(A_norm.shape[0]):
                diff = A_norm[i] - X_norm
                all_dists[i] = np.sqrt(np.sum(diff * diff))
        elif metric == 'cosine':
            for i in range(A_norm.shape[0]):
                dot_product = np.sum(A_norm[i] * X_norm)
                all_dists[i] = 1.0 - dot_product
        elif metric == 'dot':
            for i in range(A_norm.shape[0]):
                dot_product = np.sum(A_norm[i] * X_norm)
                all_dists[i] = -dot_product
        elif metric == 'manhattan':
            for i in range(A_norm.shape[0]):
                all_dists[i] = np.sum(np.abs(A_norm[i] - X_norm))
                
        # Get top K
        topk_indices = np.argsort(all_dists)[:min(K, A_norm.shape[0])]
        return topk_indices.tolist()
    
    # Merge candidates
    merged_candidates = np.concatenate(all_candidates)
    
    # Sample if too many candidates
    if len(merged_candidates) > 80000:
        # Sample without replacement
        sample_indices = np.random.choice(len(merged_candidates), 80000, replace=False)
        merged_candidates = merged_candidates[sample_indices]
    
    # Extract candidate vectors
    candidate_points = A_norm[merged_candidates]
    
    # Compute final distances
    candidate_dists = np.zeros(candidate_points.shape[0])
    
    if metric == 'l2':
        for i in range(candidate_points.shape[0]):
            diff = candidate_points[i] - X_norm
            candidate_dists[i] = np.sqrt(np.sum(diff * diff))
    elif metric == 'cosine':
        for i in range(candidate_points.shape[0]):
            dot_product = np.sum(candidate_points[i] * X_norm)
            candidate_dists[i] = 1.0 - dot_product
    elif metric == 'dot':
        for i in range(candidate_points.shape[0]):
            dot_product = np.sum(candidate_points[i] * X_norm)
            candidate_dists[i] = -dot_product
    elif metric == 'manhattan':
        for i in range(candidate_points.shape[0]):
            candidate_dists[i] = np.sum(np.abs(candidate_points[i] - X_norm))
    
    # Get final top-K
    k_to_return = min(K, len(candidate_dists))
    final_indices = np.argsort(candidate_dists)[:k_to_return]
    final_result = [merged_candidates[i] for i in final_indices]
    
    return final_result

def test_ann_for_metric(metric, dimensions, num_queries=10):
    """
    Enhanced testing function that doesn't use built-in functions
    """
    results = []
    N = 4000
    K = 10
    
    for D in dimensions:
        print(f"\nTesting {metric.upper()} ANN with dimension = {D}")
        
        # Generate random dataset
        A_np = np.random.randn(N, D).astype(np.float32)
        
        # Apply consistent normalization for cosine and dot metrics
        if metric in ('cosine', 'dot'):
            # Normalize manually
            norms = np.zeros((A_np.shape[0], 1))
            for i in range(A_np.shape[0]):
                norms[i, 0] = np.sqrt(np.sum(A_np[i] * A_np[i]))
            
            # Handle zero norms
            for i in range(A_np.shape[0]):
                if norms[i, 0] < 1e-8:
                    norms[i, 0] = 1.0
            
            # Apply normalization
            for i in range(A_np.shape[0]):
                A_np[i] = A_np[i] / norms[i, 0]
        
        # Get parameters with improved function that accounts for cluster sizes
        n_clusters, n_probe, K2 = compute_params(N, D, K, metric)
        
        # Perform offline clustering
        start_offline = time.time()
        cluster_ids, centroids = offline_kmeans(metric, A_np, n_clusters, D, N)
        torch.cuda.synchronize()
        offline_time = time.time() - start_offline
        print(f"Offline clustering time: {offline_time:.6f} s (excluded from query timing)")
        
        # Analyze actual cluster distribution
        cluster_sizes = []
        for i in range(n_clusters):
            size = sum(1 for cid in cluster_ids if cid == i)
            cluster_sizes.append(size)
        
        # Calculate statistics manually
        mean_size = sum(cluster_sizes) / len(cluster_sizes)
        sorted_sizes = sorted(cluster_sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        
        print(f"Cluster size statistics: mean={mean_size:.1f}, median={median_size:.1f}, min={min_size}, max={max_size}")
        
        if K2 > median_size:
            # Adjust K2 if it's too large compared to typical cluster size
            old_K2 = K2
            K2 = min(K2, max(int(median_size * 0.8), K))
            print(f"Adjusted K2 from {old_K2} to {K2} based on actual cluster sizes")
        
        gpu_query_times = []
        cpu_query_times = []
        gpu_recalls = []
        cpu_recalls = []
        
        for i in range(num_queries):
            # Generate and normalize query vector
            X_np = np.random.randn(D).astype(np.float32)
            if metric in ('cosine', 'dot'):
                norm_x = np.sqrt(np.sum(X_np * X_np))
                if norm_x < 1e-8:
                    norm_x = 1.0
                X_np = X_np / norm_x
            
            # Get ground truth results from exact KNN
            if metric == 'l2':
                gt_result = our_knn_l2(N, D, A_np, X_np, K)
            elif metric == 'cosine':
                gt_result = our_knn_cosine(N, D, A_np, X_np, K)
            elif metric == 'dot':
                gt_result = our_knn_dot(N, D, A_np, X_np, K)
            elif metric == 'manhattan':
                gt_result = our_knn_manhattan(N, D, A_np, X_np, K)
            
            # Run GPU ANN query with improved implementation
            torch.cuda.synchronize()
            start = time.time()
            gpu_ann_result = ann_query(metric, A_np, X_np, K, n_probe, K2, cluster_ids, centroids, D)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            gpu_query_times.append(gpu_time)
            
            # Calculate recall manually
            common_count = 0
            for idx in gpu_ann_result:
                if idx in gt_result:
                    common_count += 1
            gpu_recall = common_count / K
            gpu_recalls.append(gpu_recall)
            
            # Run CPU ANN query
            start = time.time()
            cpu_ann_result = cpu_ann_query(metric, A_np, X_np, K, n_probe, K2, cluster_ids, centroids, D)
            cpu_time = time.time() - start
            cpu_query_times.append(cpu_time)
            
            # Calculate CPU recall manually
            common_count = 0
            for idx in cpu_ann_result:
                if idx in gt_result:
                    common_count += 1
            cpu_recall = common_count / K
            cpu_recalls.append(cpu_recall)
            
            # If recall is too low for this query, log it for analysis
            if gpu_recall < 0.7:
                print(f"  Query {i}: Low GPU recall ({gpu_recall:.2f}) for {metric} metric")
        
        # Calculate averages manually
        avg_gpu_time = sum(gpu_query_times) / num_queries
        avg_cpu_time = sum(cpu_query_times) / num_queries
        avg_gpu_recall = sum(gpu_recalls) / num_queries
        avg_cpu_recall = sum(cpu_recalls) / num_queries
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
        
        # Calculate min and max recalls manually
        min_gpu_recall = gpu_recalls[0]
        max_gpu_recall = gpu_recalls[0]
        for recall in gpu_recalls:
            if recall < min_gpu_recall:
                min_gpu_recall = recall
            if recall > max_gpu_recall:
                max_gpu_recall = recall
        
        results.append({
            "dimension": D,
            "avg_gpu_query_time": avg_gpu_time,
            "avg_cpu_query_time": avg_cpu_time,
            "speedup": speedup,
            "avg_gpu_recall": avg_gpu_recall,
            "avg_cpu_recall": avg_cpu_recall,
            "offline_time": offline_time,
            "min_gpu_recall": min_gpu_recall,
            "max_gpu_recall": max_gpu_recall
        })
        
        print(f"Dimension {D}: GPU ANN time = {avg_gpu_time:.6f}s, GPU Recall = {avg_gpu_recall*100:.1f}%")
        print(f"             CPU ANN time = {avg_cpu_time:.6f}s, CPU Recall = {avg_cpu_recall*100:.1f}%")
        print(f"             GPU-CPU Speedup: {speedup:.2f}x, Offline Time = {offline_time:.6f}s")
        print(f"             GPU Recall range: {min_gpu_recall*100:.1f}% - {max_gpu_recall*100:.1f}%")
    
    return results

def main():
    """
    Main function that conducts ANN performance testing across different metrics and dimensions
    using only custom implementations from task.py
    """
    # Vector dimensions to test, from small to very large
    dimensions = [2, 8, 32, 128, 512, 2048, 8192, 32768]
    
    # Distance metrics to evaluate
    metrics = ['l2', 'cosine', 'dot', 'manhattan']
    
    # Storage for all test results
    all_results = {}
    
    # Number of queries to average for more reliable results
    num_queries = 10
    
    print("\n===== GPU Vector Search ANN Evaluation =====")
    print(f"Testing with {num_queries} queries per configuration")
    print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    
    # Run tests for each metric
    for metric in metrics:
        print(f"\n========== Testing ANN for metric: {metric.upper()} ==========")
        results = test_ann_for_metric(metric, dimensions, num_queries)
        all_results[metric] = results
    
    # Generate comprehensive performance summary
    print("\n===== PERFORMANCE SUMMARY =====")
    
    # First summarize recall rates across all metrics and dimensions
    print("\n----- RECALL RATES (%) -----")
    print("Dimension  |  L2  |  Cosine  |  Dot  |  Manhattan")
    print("----------|------|----------|-------|------------")
    
    for i, dim in enumerate(dimensions):
        if i < len(all_results['l2']):
            l2_recall = all_results['l2'][i]['avg_gpu_recall'] * 100
            cosine_recall = all_results['cosine'][i]['avg_gpu_recall'] * 100
            dot_recall = all_results['dot'][i]['avg_gpu_recall'] * 100
            manhattan_recall = all_results['manhattan'][i]['avg_gpu_recall'] * 100
            
            print(f"{dim:10d} | {l2_recall:4.1f}% | {cosine_recall:6.1f}% | {dot_recall:4.1f}% | {manhattan_recall:10.1f}%")
    
    # Then summarize GPU query times
    print("\n----- GPU QUERY TIMES (ms) -----")
    print("Dimension  |  L2  |  Cosine  |  Dot  |  Manhattan")
    print("----------|------|----------|-------|------------")
    
    for i, dim in enumerate(dimensions):
        if i < len(all_results['l2']):
            l2_time = all_results['l2'][i]['avg_gpu_query_time'] * 1000
            cosine_time = all_results['cosine'][i]['avg_gpu_query_time'] * 1000
            dot_time = all_results['dot'][i]['avg_gpu_query_time'] * 1000
            manhattan_time = all_results['manhattan'][i]['avg_gpu_query_time'] * 1000
            
            print(f"{dim:10d} | {l2_time:4.2f} | {cosine_time:6.2f} | {dot_time:4.2f} | {manhattan_time:10.2f}")
    
    # Finally summarize speedup factors
    print("\n----- GPU-CPU SPEEDUP (×) -----")
    print("Dimension  |  L2  |  Cosine  |  Dot  |  Manhattan")
    print("----------|------|----------|-------|------------")
    
    for i, dim in enumerate(dimensions):
        if i < len(all_results['l2']):
            l2_speedup = all_results['l2'][i]['speedup']
            cosine_speedup = all_results['cosine'][i]['speedup']
            dot_speedup = all_results['dot'][i]['speedup']
            manhattan_speedup = all_results['manhattan'][i]['speedup']
            
            print(f"{dim:10d} | {l2_speedup:4.1f}× | {cosine_speedup:6.1f}× | {dot_speedup:4.1f}× | {manhattan_speedup:10.1f}×")
    
    # Print key observations
    print("\n===== KEY FINDINGS =====")
    
    # Find best recall rate for each metric manually
    best_l2_recall = 0
    best_cosine_recall = 0
    best_dot_recall = 0
    best_manhattan_recall = 0
    
    for r in all_results['l2']:
        if r['avg_gpu_recall'] > best_l2_recall:
            best_l2_recall = r['avg_gpu_recall']
    
    for r in all_results['cosine']:
        if r['avg_gpu_recall'] > best_cosine_recall:
            best_cosine_recall = r['avg_gpu_recall']
    
    for r in all_results['dot']:
        if r['avg_gpu_recall'] > best_dot_recall:
            best_dot_recall = r['avg_gpu_recall']
    
    for r in all_results['manhattan']:
        if r['avg_gpu_recall'] > best_manhattan_recall:
            best_manhattan_recall = r['avg_gpu_recall']
    
    print(f"1. Best recall rates achieved: L2: {best_l2_recall*100:.1f}%, Cosine: {best_cosine_recall*100:.1f}%, "
          f"Dot: {best_dot_recall*100:.1f}%, Manhattan: {best_manhattan_recall*100:.1f}%")
    
    # Find best speedup for each metric manually
    best_l2_speedup = 0
    best_cosine_speedup = 0
    best_dot_speedup = 0
    best_manhattan_speedup = 0
    
    for r in all_results['l2']:
        if r['speedup'] > best_l2_speedup:
            best_l2_speedup = r['speedup']
    
    for r in all_results['cosine']:
        if r['speedup'] > best_cosine_speedup:
            best_cosine_speedup = r['speedup']
    
    for r in all_results['dot']:
        if r['speedup'] > best_dot_speedup:
            best_dot_speedup = r['speedup']
    
    for r in all_results['manhattan']:
        if r['speedup'] > best_manhattan_speedup:
            best_manhattan_speedup = r['speedup']
    
    print(f"2. Best GPU-CPU speedups achieved: L2: {best_l2_speedup:.1f}×, Cosine: {best_cosine_speedup:.1f}×, "
          f"Dot: {best_dot_speedup:.1f}×, Manhattan: {best_manhattan_speedup:.1f}×")
    
    # Calculate average recall across all dimensions for each metric manually
    avg_l2_recall = sum(r['avg_gpu_recall'] for r in all_results['l2']) / len(all_results['l2'])
    avg_cosine_recall = sum(r['avg_gpu_recall'] for r in all_results['cosine']) / len(all_results['cosine'])
    avg_dot_recall = sum(r['avg_gpu_recall'] for r in all_results['dot']) / len(all_results['dot'])
    avg_manhattan_recall = sum(r['avg_gpu_recall'] for r in all_results['manhattan']) / len(all_results['manhattan'])
    
    print(f"3. Average recall across all dimensions: L2: {avg_l2_recall*100:.1f}%, Cosine: {avg_cosine_recall*100:.1f}%, "
          f"Dot: {avg_dot_recall*100:.1f}%, Manhattan: {avg_manhattan_recall*100:.1f}%")
    
    # Note about high-dimensional performance
    high_dim_l2_recall = all_results['l2'][-1]['avg_gpu_recall'] * 100
    high_dim_cosine_recall = all_results['cosine'][-1]['avg_gpu_recall'] * 100
    
    print(f"4. For highest dimension tested ({dimensions[-1]}D): L2 recall: {high_dim_l2_recall:.1f}%, "
          f"Cosine recall: {high_dim_cosine_recall:.1f}%")
    
    print("\nTest completed successfully. All metrics evaluated across all dimensions.")

if __name__ == "__main__":
    main()
