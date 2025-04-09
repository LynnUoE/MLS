import triton
import triton.language as tl
import torch
import time
import numpy as np
from test import testdata_knn, testdata_kmeans, testdata_ann

# -----------------------------------------------------------------------------
# Generalized Triton kernels for distance metrics
# -----------------------------------------------------------------------------
@triton.jit
def manhattan_distance_kernel(A, X, output, D: tl.constexpr, stride_A: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        diff = a - x
        acc += tl.sum(tl.abs(diff))
    tl.store(output + pid, acc)

@triton.jit
def dot_distance_kernel(A, X, output, D: tl.constexpr, stride_A: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        a = tl.load(row_ptr + offs, mask=offs < D, other=0.0)
        x = tl.load(X + offs, mask=offs < D, other=0.0)
        acc += tl.sum(a * x)
    tl.store(output + pid, -acc)

@triton.jit
def cosine_distance_kernel(A, X, output, D: tl.constexpr, stride_A: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    dot = 0.0
    norm_a = 0.0
    norm_x = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        dot += tl.sum(a * x)
        norm_a += tl.sum(a * a)
        norm_x += tl.sum(x * x)
    
    # Improved numerical stability with epsilon
    norm_a = tl.maximum(norm_a, 1e-8)
    norm_x = tl.maximum(norm_x, 1e-8)
    norm_product = tl.sqrt(norm_a) * tl.sqrt(norm_x)
    
    # Clamp similarity to valid range [-1, 1]
    sim = tl.minimum(tl.maximum(dot / norm_product, -1.0), 1.0)
    tl.store(output + pid, 1.0 - sim)

@triton.jit
def l2_distance_kernel(A, X, output, D: tl.constexpr, stride_A: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        a = tl.load(row_ptr + offs, mask=offs < D, other=0.0)
        x = tl.load(X + offs, mask=offs < D, other=0.0)
        diff = a - x
        acc += tl.sum(diff * diff)
    tl.store(output + pid, tl.sqrt(acc))

@triton.jit
def topk_kernel(distances, indices, output_indices, N, K, BLOCK_SIZE):
    pid = tl.program_id(0)
    if pid == 0:
        top_distances = tl.zeros([K], dtype=tl.float32) + float('inf')
        top_indices = tl.zeros([K], dtype=tl.int32) - 1
        for i in range(N):
            dist = tl.load(distances + i)
            idx = tl.load(indices + i)
            for k in range(K):
                if dist < top_distances[k]:
                    for j in range(K-1, k, -1):
                        top_distances[j] = top_distances[j-1]
                        top_indices[j] = top_indices[j-1]
                    top_distances[k] = dist
                    top_indices[k] = idx
                    break
        for k in range(K):
            tl.store(output_indices + k, top_indices[k])

def gpu_topk(distances, k):
    N = distances.shape[0]
    k = min(k, N)
    temp_distances = distances.clone()
    indices = torch.arange(0, N, device=distances.device)
    result_indices = torch.empty(k, device=distances.device, dtype=torch.int64)
    for i in range(k):
        min_idx = torch.argmin(temp_distances)
        result_indices[i] = indices[min_idx]
        temp_distances[min_idx] = float('inf')
    return result_indices

# -----------------------------------------------------------------------------
# Helper functions for GPU distance computations using Triton kernels
# -----------------------------------------------------------------------------
def gpu_l2_distance(A, x, D, block_size=None):
    if block_size is None:
        block_size = 256 if D > 512 else 128
    N = A.shape[0]
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    l2_distance_kernel[grid](A, x.unsqueeze(0), distances, D, A.stride(0), BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    return distances

def gpu_cosine_distance(A, x, D, block_size=None):
    if block_size is None:
        block_size = 256 if D > 512 else 128
    N = A.shape[0]
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    cosine_distance_kernel[grid](A, x.unsqueeze(0), distances, D, A.stride(0), BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    return distances

def gpu_dot_distance(A, x, D, block_size=None):
    if block_size is None:
        block_size = 256 if D > 512 else 128
    N = A.shape[0]
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    dot_distance_kernel[grid](A, x.unsqueeze(0), distances, D, A.stride(0), BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    return distances

def gpu_manhattan_distance(A, x, D, block_size=None):
    if block_size is None:
        block_size = 256 if D > 512 else 128
    N = A.shape[0]
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    manhattan_distance_kernel[grid](A, x.unsqueeze(0), distances, D, A.stride(0), BLOCK_SIZE=block_size)
    torch.cuda.synchronize()
    return distances

# -----------------------------------------------------------------------------
# KMeans implementations with different distance metrics using GPU kernels
# -----------------------------------------------------------------------------
def kmeans_plus_plus_l2(A, K):
    N = A.shape[0]
    centroids = []
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    D = A.shape[1]
    for _ in range(1, K):
        # Use the GPU kernel version for L2 distances
        distances = gpu_l2_distance(A, centroids[0], D) ** 2
        for c in centroids[1:]:
            d_new = gpu_l2_distance(A, c, D) ** 2
            distances = torch.min(distances, d_new)
        probs = distances / distances.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=A.device)
        next_idx = torch.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx].squeeze(0))
    return torch.stack(centroids)

def our_kmeans_l2(N, D, A_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    centroids = kmeans_plus_plus_l2(A, K)
    max_iter = 100
    tol = 1e-4
    for i in range(max_iter):
        distances = torch.zeros((N, K), device=A.device)
        for j in range(K):
            d = gpu_l2_distance(A, centroids[j], D)
            distances[:, j] = d
        cluster_ids = torch.argmin(distances, dim=1)
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.size(0) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[j])
        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return cluster_ids.tolist(), centroids.cpu()

def kmeans_plus_plus_cosine(A, K):
    N = A.shape[0]
    centroids = []
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    D = A.shape[1]
    for _ in range(1, K):
        distances = gpu_cosine_distance(A, centroids[0], D) ** 2
        for c in centroids[1:]:
            d_new = gpu_cosine_distance(A, c, D) ** 2
            distances = torch.min(distances, d_new)
        probs = distances / distances.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=A.device)
        next_idx = torch.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx].squeeze(0))
    return torch.stack(centroids)

def our_kmeans_cosine(N, D, A_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    centroids = kmeans_plus_plus_cosine(A, K)
    max_iter = 100
    tol = 1e-4
    for i in range(max_iter):
        distances = torch.zeros((N, K), device=A.device)
        for j in range(K):
            d = gpu_cosine_distance(A, centroids[j], D)
            distances[:, j] = d
        cluster_ids = torch.argmin(distances, dim=1)
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.size(0) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[j])
        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return cluster_ids.tolist(), centroids.cpu()

def kmeans_plus_plus_dot(A, K):
    N = A.shape[0]
    centroids = []
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    D = A.shape[1]
    for _ in range(1, K):
        distances = gpu_dot_distance(A, centroids[0], D) ** 2
        for c in centroids[1:]:
            d_new = gpu_dot_distance(A, c, D) ** 2
            distances = torch.min(distances, d_new)
        probs = distances / distances.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=A.device)
        next_idx = torch.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx].squeeze(0))
    return torch.stack(centroids)

def our_kmeans_dot(N, D, A_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    centroids = kmeans_plus_plus_dot(A, K)
    max_iter = 100
    tol = 1e-4
    for i in range(max_iter):
        distances = torch.zeros((N, K), device=A.device)
        for j in range(K):
            d = gpu_dot_distance(A, centroids[j], D)
            distances[:, j] = d
        cluster_ids = torch.argmin(distances, dim=1)
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.size(0) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[j])
        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return cluster_ids.tolist(), centroids.cpu()

def kmeans_plus_plus_manhattan(A, K):
    N = A.shape[0]
    centroids = []
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    D = A.shape[1]
    for _ in range(1, K):
        distances = gpu_manhattan_distance(A, centroids[0], D) ** 2
        for c in centroids[1:]:
            d_new = gpu_manhattan_distance(A, c, D) ** 2
            distances = torch.min(distances, d_new)
        probs = distances / distances.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=A.device)
        next_idx = torch.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx].squeeze(0))
    return torch.stack(centroids)

def our_kmeans_manhattan(N, D, A_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    centroids = kmeans_plus_plus_manhattan(A, K)
    max_iter = 100
    tol = 1e-4
    for i in range(max_iter):
        distances = torch.zeros((N, K), device=A.device)
        for j in range(K):
            d = gpu_manhattan_distance(A, centroids[j], D)
            distances[:, j] = d
        cluster_ids = torch.argmin(distances, dim=1)
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.size(0) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[j])
        new_centroids = torch.stack(new_centroids)
        if torch.norm(new_centroids - centroids) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids
    return cluster_ids.tolist(), centroids.cpu()

# -----------------------------------------------------------------------------
# ANN implementations with different distance metrics using precomputed centroids
# -----------------------------------------------------------------------------
def our_ann_l2(N, D, A_np, X_np, K):
    # Set parameters (these may be adjusted externally)
    if N <= 1000:
        K1 = min(int(N/10), 20)
    else:
        K1 = min(int(np.sqrt(N)), 100)
    if D > 100:
        K1 = max(int(K1 * 0.7), 10)
    K1 = max(K1, 2*K)
    K2 = max(K, 20)
    
    cluster_ids, centroids = our_kmeans_l2(N, D, A_np, K1)
    # Note: now the centroids are precomputed from the offline phase.
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    cluster_ids = torch.tensor(cluster_ids, device="cuda")
    centroids = centroids.to("cuda")
    
    centroid_distances = gpu_l2_distance(centroids, X, D)
    nearest_clusters = gpu_topk(centroid_distances, min(K1, centroids.size(0)))
    
    all_candidate_indices = []
    for cluster_idx in nearest_clusters:
        idx = cluster_idx.item()
        cluster_point_indices = (cluster_ids == idx).nonzero(as_tuple=True)[0]
        if cluster_point_indices.size(0) > 0:
            cluster_points = A[cluster_point_indices]
            distances = gpu_l2_distance(cluster_points, X, D)
            actual_K2 = min(K2, cluster_point_indices.size(0))
            if actual_K2 > 0:
                topk_indices = gpu_topk(distances, actual_K2)
                selected_indices = cluster_point_indices[topk_indices]
                all_candidate_indices.append(selected_indices)
    if not all_candidate_indices:
        return our_knn_l2(N, D, A_np, X_np, K)
    merged_candidates = torch.cat(all_candidate_indices).cpu().numpy()
    candidate_vectors = A_np[merged_candidates]
    topk_local_indices = our_knn_l2(len(merged_candidates), D, candidate_vectors, X_np, min(K, len(merged_candidates)))
    final_indices = [merged_candidates[idx] for idx in topk_local_indices]
    return final_indices

def our_ann_cosine(N, D, A_np, X_np, K):
    # Ensure proper normalization of input vectors
    A_norm = A_np / (np.linalg.norm(A_np, axis=1, keepdims=True) + 1e-8)
    X_norm = X_np / (np.linalg.norm(X_np) + 1e-8)
    
    # Adjust K1 and K2 parameters based on dimensionality
    if N <= 1000:
        K1 = min(int(N/10), 20)
    else:
        K1 = min(int(np.sqrt(N)), 100)
    
    # Increase cluster probing for high dimensions
    if D > 512:
        K1 = max(int(K1 * 1.5), 30)  # Increase clusters for high dimensions
    
    K1 = max(K1, 3*K)  # Ensure we have enough clusters
    K2 = max(K * 3, 50)  # Increase candidates per cluster
    
    # Run clustering on normalized vectors
    cluster_ids, centroids = our_kmeans_cosine(N, D, A_norm, K1)
    
    A = torch.tensor(A_norm, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_norm, dtype=torch.float32, device="cuda")
    cluster_ids = torch.tensor(cluster_ids, device="cuda")
    centroids = centroids.to("cuda")
    
    # Compute cosine distances to centroids
    centroid_distances = gpu_cosine_distance(centroids, X, D)
    
    # Increase the number of clusters we probe in high dimensions
    n_probe = min(max(int(K1 * 0.8), K*2), K1) if D > 512 else min(K1, centroids.size(0))
    nearest_clusters = gpu_topk(centroid_distances, n_probe)
    
    all_candidate_indices = []
    for cluster_idx in nearest_clusters:
        idx = cluster_idx.item()
        cluster_point_indices = (cluster_ids == idx).nonzero(as_tuple=True)[0]
        if cluster_point_indices.size(0) > 0:
            cluster_points = A[cluster_point_indices]
            distances = gpu_cosine_distance(cluster_points, X, D)
            actual_K2 = min(K2, cluster_point_indices.size(0))
            if actual_K2 > 0:
                topk_indices = gpu_topk(distances, actual_K2)
                selected_indices = cluster_point_indices[topk_indices]
                all_candidate_indices.append(selected_indices)

    if not all_candidate_indices:
        return our_knn_cosine(N, D, A_norm, X_norm, K)
    
    merged_candidates = torch.cat(all_candidate_indices).cpu().numpy()
    candidate_vectors = A_norm[merged_candidates]
    topk_local_indices = our_knn_cosine(len(merged_candidates), D, candidate_vectors, X_norm, min(K, len(merged_candidates)))
    final_indices = [merged_candidates[idx] for idx in topk_local_indices]
    return final_indices

def our_ann_dot(N, D, A_np, X_np, K):
    if N <= 1000:
        K1 = min(int(N/10), 20)
    else:
        K1 = min(int(np.sqrt(N)), 100)
    if D > 100:
        K1 = max(int(K1 * 0.7), 10)
    K1 = max(K1, 2*K)
    K2 = max(K, 20)
    
    cluster_ids, centroids = our_kmeans_dot(N, D, A_np, K1)
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    cluster_ids = torch.tensor(cluster_ids, device="cuda")
    centroids = centroids.to("cuda")
    
    centroid_distances = gpu_dot_distance(centroids, X, D)
    nearest_clusters = gpu_topk(centroid_distances, min(K1, centroids.size(0)))
    
    all_candidate_indices = []
    for cluster_idx in nearest_clusters:
        idx = cluster_idx.item()
        cluster_point_indices = (cluster_ids == idx).nonzero(as_tuple=True)[0]
        if cluster_point_indices.size(0) > 0:
            cluster_points = A[cluster_point_indices]
            distances = gpu_dot_distance(cluster_points, X, D)
            actual_K2 = min(K2, cluster_point_indices.size(0))
            if actual_K2 > 0:
                topk_indices = gpu_topk(distances, actual_K2)
                selected_indices = cluster_point_indices[topk_indices]
                all_candidate_indices.append(selected_indices)
    if not all_candidate_indices:
        return our_knn_dot(N, D, A_np, X_np, K)
    merged_candidates = torch.cat(all_candidate_indices).cpu().numpy()
    candidate_vectors = A_np[merged_candidates]
    topk_local_indices = our_knn_dot(len(merged_candidates), D, candidate_vectors, X_np, min(K, len(merged_candidates)))
    final_indices = [merged_candidates[idx] for idx in topk_local_indices]
    return final_indices

def our_ann_manhattan(N, D, A_np, X_np, K):
    if N <= 1000:
        K1 = min(int(N/10), 20)
    else:
        K1 = min(int(np.sqrt(N)), 100)
    if D > 100:
        K1 = max(int(K1 * 0.7), 10)
    K1 = max(K1, 2*K)
    K2 = max(K, 20)
    
    cluster_ids, centroids = our_kmeans_manhattan(N, D, A_np, K1)
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    cluster_ids = torch.tensor(cluster_ids, device="cuda")
    centroids = centroids.to("cuda")
    
    centroid_distances = gpu_manhattan_distance(centroids, X, D)
    nearest_clusters = gpu_topk(centroid_distances, min(K1, centroids.size(0)))
    
    all_candidate_indices = []
    for cluster_idx in nearest_clusters:
        idx = cluster_idx.item()
        cluster_point_indices = (cluster_ids == idx).nonzero(as_tuple=True)[0]
        if cluster_point_indices.size(0) > 0:
            cluster_points = A[cluster_point_indices]
            distances = gpu_manhattan_distance(cluster_points, X, D)
            actual_K2 = min(K2, cluster_point_indices.size(0))
            if actual_K2 > 0:
                topk_indices = gpu_topk(distances, actual_K2)
                selected_indices = cluster_point_indices[topk_indices]
                all_candidate_indices.append(selected_indices)
    if not all_candidate_indices:
        return our_knn_manhattan(N, D, A_np, X_np, K)
    merged_candidates = torch.cat(all_candidate_indices).cpu().numpy()
    candidate_vectors = A_np[merged_candidates]
    topk_local_indices = our_knn_manhattan(len(merged_candidates), D, candidate_vectors, X_np, min(K, len(merged_candidates)))
    final_indices = [merged_candidates[idx] for idx in topk_local_indices]
    return final_indices

# -----------------------------------------------------------------------------
# KNN implementations using fixed GPU TopK
# -----------------------------------------------------------------------------
def our_knn_l2(N, D, A_np, X_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    block_size = 256 if D > 512 else 128
    l2_distance_kernel[grid](A, X, distances, D, A.stride(0), BLOCK_SIZE=block_size)
    topk_indices = gpu_topk(distances, K)
    return topk_indices.cpu().numpy().tolist()

def our_knn_cosine(N, D, A_np, X_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    block_size = 256 if D > 512 else 128
    cosine_distance_kernel[grid](A, X, distances, D, A.stride(0), BLOCK_SIZE=block_size)
    topk_indices = gpu_topk(distances, K)
    return topk_indices.cpu().numpy().tolist()

def our_knn_dot(N, D, A_np, X_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    block_size = 256 if D > 512 else 128
    dot_distance_kernel[grid](A, X, distances, D, A.stride(0), BLOCK_SIZE=block_size)
    topk_indices = gpu_topk(distances, K)
    return topk_indices.cpu().numpy().tolist()

def our_knn_manhattan(N, D, A_np, X_np, K):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    distances = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    block_size = 256 if D > 512 else 128
    manhattan_distance_kernel[grid](A, X, distances, D, A.stride(0), BLOCK_SIZE=block_size)
    topk_indices = gpu_topk(distances, K)
    return topk_indices.cpu().numpy().tolist()
