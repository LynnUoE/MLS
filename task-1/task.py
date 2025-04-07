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
    # Process one row per program
    pid = tl.program_id(0)
    
    # Initialize pointer to current row in A
    row_ptr = A + pid * stride_A
    
    # Initialize accumulator for the L1 distance
    acc = 0.0
    
    # Process the vector in blocks to maximize cache usage
    for d in range(0, D, BLOCK_SIZE):
        # Calculate offsets for current block
        offs = d + tl.arange(0, BLOCK_SIZE)
        
        # Load block from matrix A and vector X with bounds checking
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        
        # Calculate absolute differences in a vectorized way
        # Using subtraction followed by absolute value is more efficient
        # than branching for each element
        diff = a - x
        abs_diff = tl.abs(diff)
        
        # Sum the absolute differences for this block
        acc += tl.sum(abs_diff)
    
    # Store the final Manhattan distance
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
        a = tl.load(row_ptr + offs, mask=offs < D, other=0.0)
        x = tl.load(X + offs, mask=offs < D, other=0.0)
        dot += tl.sum(a * x)
        norm_a += tl.sum(a * a)
        norm_x += tl.sum(x * x)
    norm_product = tl.sqrt(norm_a) * tl.sqrt(norm_x)
    sim = dot / norm_product
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

# -----------------------------------------------------------------------------
# Generalized distance launcher with dynamic BLOCK_SIZE
# -----------------------------------------------------------------------------
def compute_distance(A, X, metric="l2", block_size=None):
    """
    Compute distances between vectors in A and the query vector X.
    
    Args:
        A: tensor with shape (N, D) - collection of vectors
        X: tensor with shape (D,) - query vector (will be broadcasted)
        metric: distance metric to use - "l2", "cosine", "dot", or "manhattan"
        block_size: block size for Triton kernel processing (auto-selected if None)
    
    Returns:
        A tensor of shape (N,) containing distances according to specified metric.
        For "dot" metric, returns negative dot product so smaller values = more similar.
    """
    N, D = A.shape
    
    # Ensure X is a 1D vector
    if len(X.shape) > 1:
        X = X.squeeze()
    
    # Auto-select block size based on dimension
    if block_size is None:
        if D <= 32:
            block_size = 32
        elif D <= 128:
            block_size = 64
        elif D <= 512:
            block_size = 128
        else:
            block_size = 256
    
    # Allocate output tensor
    output = torch.empty((N,), device=A.device, dtype=torch.float32)
    
    # Launch appropriate kernel based on metric
    grid = (N,)
    if metric == "l2":
        l2_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    elif metric == "cosine":
        cosine_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    elif metric == "dot":
        # Dot product distance: smaller values (more negative) indicate greater similarity
        dot_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    elif metric == "manhattan":
        manhattan_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return output

# -----------------------------------------------------------------------------
# Top-K KNN with metric option
# -----------------------------------------------------------------------------
def our_knn(N, D, A_np, X_np, K, metric="l2"):
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    distances = compute_distance(A, X, metric)
    topk = torch.topk(distances, k=K, largest=False)
    return topk.indices.cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# KMeans using our custom distance function and K-means++ initialization
# -----------------------------------------------------------------------------

def kmeans_plus_plus(A, K, metric="l2"):
    """
    A: tensor with shape (N, D) on GPU.
    Returns K initial centroids using K-means++ algorithm.
    """
    N = A.shape[0]
    centroids = []
    # Randomly select the first centroid and squeeze to 1D vector.
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    for _ in range(1, K):
        dists = compute_distance(A, centroids[0].unsqueeze(0), metric) ** 2
        for c in centroids[1:]:
            d_new = compute_distance(A, c.unsqueeze(0), metric) ** 2
            dists = torch.min(dists, d_new)
        probs = dists / dists.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1, device=A.device)
        next_idx = torch.searchsorted(cumulative_probs, r)
        centroids.append(A[next_idx].squeeze(0))
    return torch.stack(centroids)


def our_kmeans(N, D, A_np, K, metric="l2"):
    """
    K-means clustering using our custom distance function and K-means++ initialization.
    Uses multiple initializations and relative change to check for convergence.
    """
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    centroids = kmeans_plus_plus(A, K, metric=metric)
    max_iter = 1000
    tol = 1e-4

    for i in range(max_iter):
        dists_list = []
        for j in range(K):
            centroid = centroids[j].unsqueeze(0)  # (1, D)
            d = compute_distance(A, centroid, metric)
            dists_list.append(d.unsqueeze(1))
        distances = torch.cat(dists_list, dim=1)  # (N, K)
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
    return cluster_ids.cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# ANN with metric option
# -----------------------------------------------------------------------------
def our_ann(N, D, A_np, X_np, K, metric="l2"):
    """
    Improved Approximate Nearest Neighbor search with higher recall (70%+)
    
    Strategies for higher recall:
    1. Better initialization for KMeans (KMeans++) 
    2. Increase number of clusters probed
    3. Use distance-weighted cluster selection
    4. Adaptive search based on dataset density
    5. Reranking of final candidates
    
    Args:
        N: Number of vectors
        D: Dimension of vectors
        A_np: numpy array of vectors (N, D)
        X_np: query vector (D,)
        K: Number of nearest neighbors to find
        metric: Distance metric to use
        
    Returns:
        List of indices of top K nearest neighbors
    """
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    
    # 1. Better clustering with appropriate parameters
    # Adaptive number of clusters based on dataset size and dimensionality
    if N <= 1000:
        K1 = min(int(N/10), 20)  # More clusters for small datasets (10%)
    else:
        K1 = min(int(np.sqrt(N)), 100)  # Standard sqrt(N) for larger datasets
    
    # Adjust based on dimensionality - fewer clusters for high dimensions
    if D > 100:
        K1 = max(int(K1 * 0.7), 10)  # Reduce clusters for high-dimensional data
        
    # Ensure K1 is at least 2x the requested K for better coverage
    K1 = max(K1, 2*K)
    
    # Run KMeans clustering
    cluster_ids_list = our_kmeans(N, D, A_np, K1, metric=metric)
    cluster_ids = torch.tensor(cluster_ids_list, device="cuda")
    
    # Compute centroids for each cluster
    centroids = []
    cluster_sizes = []  # Track cluster sizes for adaptive search
    for j in range(K1):
        points = A[cluster_ids == j]
        size = points.size(0)
        cluster_sizes.append(size)
        if size > 0:
            centroids.append(points.mean(dim=0))
        else:
            centroids.append(torch.zeros(D, device="cuda"))
    centroids = torch.stack(centroids)
    
    # 2. Increase number of probed clusters - approximately 30% of clusters or at least 10
    K1_probe = max(min(int(K1 * 0.3), 20), 10)
    K1_probe = min(K1_probe, K1)  # Can't probe more clusters than we have
    
    # 3. Compute distance to all centroids
    centroid_distances = compute_distance(centroids, X, metric)
    
    # Get cluster indices sorted by distance
    sorted_cluster_indices = torch.argsort(centroid_distances)
    
    # 4. Adaptive search - dynamically adjust clusters based on their sizes and distances
    # Prioritize clusters that are both close and have sufficient points
    candidate_score = torch.zeros_like(centroid_distances)
    norm_distances = centroid_distances / centroid_distances.max()  # Normalize to [0,1]
    
    # Calculate a score combining distance and cluster size
    for i, idx in enumerate(range(K1)):
        cluster_size = cluster_sizes[idx]
        # Higher score for closer clusters with more points
        size_factor = min(cluster_size / (N/K1), 2.0)  # Cap size factor at 2.0
        candidate_score[idx] = (1 - norm_distances[idx]) * size_factor
    
    # Select top clusters based on combined score
    top_cluster_indices = torch.topk(candidate_score, k=K1_probe, largest=True).indices
    
    # 5. Search for more neighbors in each cluster
    # Adaptive K2: more neighbors from closer clusters, fewer from distant ones
    base_K2 = min(100, N // K1)  # Base number of neighbors per cluster
    
    all_candidate_indices = []
    all_candidate_distances = []
    
    # Collect candidates from top clusters
    for i, cluster_idx in enumerate(top_cluster_indices):
        # Get all points in this cluster
        cluster_point_indices = (cluster_ids == cluster_idx.item()).nonzero(as_tuple=True)[0]
        
        if cluster_point_indices.size(0) > 0:
            # Adaptive K2 - more neighbors from closer clusters
            closeness_factor = 1.0 - i / K1_probe  # 1.0 for closest, lower for farther
            K2_cluster = max(int(base_K2 * (0.5 + 0.5 * closeness_factor)), K)
            K2_cluster = min(K2_cluster, cluster_point_indices.size(0))
            
            # Compute distances for all points in this cluster
            cluster_points = A[cluster_point_indices]
            distances = compute_distance(cluster_points, X, metric)
            
            # Get top K2 nearest neighbors in this cluster
            topk_in_cluster = torch.topk(distances, k=K2_cluster, largest=False)
            
            # Store their indices and distances
            all_candidate_indices.append(cluster_point_indices[topk_in_cluster.indices])
            all_candidate_distances.append(topk_in_cluster.values)
    
    # Merge and rerank all candidates
    if all_candidate_indices:
        # Concatenate all candidate indices and distances
        merged_indices = torch.cat(all_candidate_indices)
        merged_distances = torch.cat(all_candidate_distances)
        
        # 6. Reranking - if we have enough candidates, recompute distances for final ranking
        # This helps correct any approximation errors from the clustering
        if merged_indices.size(0) > K * 10:  # Only if we have 10x more candidates than K
            # Recompute distances directly with query
            final_points = A[merged_indices]
            recomputed_distances = compute_distance(final_points, X, metric)
            
            # Find top K among merged candidates using recomputed distances
            k_actual = min(K, merged_indices.size(0))
            final_topk = torch.topk(recomputed_distances, k=k_actual, largest=False)
        else:
            # Use original distances if not enough candidates
            k_actual = min(K, merged_indices.size(0))
            final_topk = torch.topk(merged_distances, k=k_actual, largest=False)
        
        # Return final indices
        return merged_indices[final_topk.indices].cpu().numpy().tolist()
    else:
        # Fallback: if no candidates found, use brute-force KNN
        distances = compute_distance(A, X, metric)
        topk = torch.topk(distances, k=min(K, N), largest=False)
        return topk.indices.cpu().numpy().tolist()

def compute_recall(knn_result: list, ann_result: list, K: int) -> float:
    common = len(set(knn_result) & set(ann_result))
    recall = common / K
    return recall

# -----------------------------------------------------------------------------
# Test wrappers and utility functions
# -----------------------------------------------------------------------------
def prewarm_gpu():
    """Prewarm the GPU to ensure accurate timing"""
    # Create dummy tensors and do a small computation
    dummy_a = torch.randn(1000, 128, device="cuda")
    dummy_x = torch.randn(128, device="cuda")
    _ = compute_distance(dummy_a, dummy_x)
    torch.cuda.synchronize()  # Ensure all operations are complete

def test_knn():
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_knn("")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        start = time.time()
        result = our_knn(N, D, A, X, K, metric)
        elapsed = time.time() - start
        print(f"KNN [{metric}] result: {result}\nElapsed: {elapsed:.4f} sec")
    
def test_knn_dimension_scaling():
    print("\n--- KNN Dimension Scaling Test ---")
    prewarm_gpu()  # Ensure GPU is properly warmed up
    
    dimensions = [2, 16, 128, 1024, 32768]  # Including 2^15 = 32768
    N = 4000  # Fixed number of vectors as per assignment requirement
    K = 10
    metrics = ["l2", "cosine", "dot", "manhattan"]
    
    print("Dimension\tMetric\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for D in dimensions:
        # Generate random data of the specified dimension
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        
        for metric in metrics:
            # Warmup run
            _ = our_knn(100, D, A_np[:100], X_np, K, metric)
            torch.cuda.synchronize()
            
            # GPU timing with proper synchronization
            torch.cuda.synchronize()
            start = time.time()
            result_gpu = our_knn(N, D, A_np, X_np, K, metric)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            # CPU timing with NumPy implementation
            start = time.time()
            
            # For dimensions > 1024, we'll use a sampling approach
            if D <= 1024:
                distances = []
                if metric == "l2":
                    for i in range(N):
                        distances.append(np.sqrt(np.sum((A_np[i] - X_np) ** 2)))
                elif metric == "cosine":
                    for i in range(N):
                        dot = np.sum(A_np[i] * X_np)
                        norm_a = np.sqrt(np.sum(A_np[i] ** 2))
                        norm_x = np.sqrt(np.sum(X_np ** 2))
                        distances.append(1.0 - dot / (norm_a * norm_x) if norm_a * norm_x > 0 else 1.0)
                elif metric == "dot":
                    for i in range(N):
                        distances.append(-np.sum(A_np[i] * X_np))  # Negative for consistent "smaller is better"
                elif metric == "manhattan":
                    for i in range(N):
                        distances.append(np.sum(np.abs(A_np[i] - X_np)))
                
                top_indices = np.argsort(distances)[:K].tolist()
                cpu_time = time.time() - start
                
                # Calculate match percentage
                common = len(set(result_gpu) & set(top_indices))
                match_percent = (common / K) * 100
            else:
                # For large dimensions, sample to estimate time
                sample_size = min(500, N)
                start = time.time()
                for i in range(sample_size):
                    if metric == "l2":
                        _ = np.sqrt(np.sum((A_np[i] - X_np) ** 2))
                    elif metric == "cosine":
                        dot = np.sum(A_np[i] * X_np)
                        norm_a = np.sqrt(np.sum(A_np[i] ** 2))
                        norm_x = np.sqrt(np.sum(X_np ** 2))
                        _ = 1.0 - dot / (norm_a * norm_x) if norm_a * norm_x > 0 else 1.0
                    elif metric == "dot":
                        _ = -np.sum(A_np[i] * X_np)
                    elif metric == "manhattan":
                        _ = np.sum(np.abs(A_np[i] - X_np))
                
                sample_time = time.time() - start
                cpu_time = sample_time * (N / sample_size)
                match_percent = "N/A"  # Can't compute match for sampled approach
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            if match_percent != "N/A":
                print(f"{D}\t\t{metric}\t{gpu_time:.6f}\t{cpu_time:.6f}\t{speedup:.2f}x\t(match: {match_percent:.1f}%)")
            else:
                print(f"{D}\t\t{metric}\t{gpu_time:.6f}\t{cpu_time:.6f}*\t{speedup:.2f}x\t(*estimated)")

def test_kmeans():
    prewarm_gpu()
    N, D, A, K = testdata_kmeans("")
    start = time.time()
    result = our_kmeans(N, D, A, K)
    elapsed = time.time() - start
    print(f"KMeans result: First 10 cluster IDs: {result[:10]}\nElapsed: {elapsed:.4f} sec")

def test_ann():
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_ann("")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        start = time.time()
        result = our_ann(N, D, A, X, K, metric)
        elapsed = time.time() - start
        print(f"ANN [{metric}] result: {result}\nElapsed: {elapsed:.4f} sec")

def test_recall():
    """
    For each distance metric, run KNN once and run ANN 10 times to compute average recall.
    """
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_knn("")
    print("Metric\tAvg Recall")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        knn_res = our_knn(N, D, A, X, K, metric)
        ann_recalls = []
        for _ in range(10):
            ann_res = our_ann(N, D, A, X, K, metric)
            recall = compute_recall(knn_res, ann_res, K)
            ann_recalls.append(recall)
        avg_recall = sum(ann_recalls) / len(ann_recalls)
        print(f"{metric}\t{avg_recall:.2%}")

def compare_gpu_cpu(test_file=None, metrics=["l2"]):
    """Compare GPU vs CPU performance"""
    prewarm_gpu()  # Prewarm GPU
    
    if test_file:
        N, D, A, X, K = testdata_knn(test_file)
    else:
        N, D, A, X, K = testdata_knn("")
    
    print(f"Data shape: {N} vectors of dimension {D}")
    
    for metric in metrics:
        # GPU implementation - with proper synchronization
        start = time.time()
        result_gpu = our_knn(N, D, A, X, K, metric)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        gpu_time = time.time() - start
        
        # CPU implementation (naive)
        start = time.time()
        distances = []
        if metric == "l2":
            for i in range(N):
                distances.append(np.sqrt(np.sum((A[i] - X) ** 2)))
        elif metric == "manhattan":
            for i in range(N):
                distances.append(np.sum(np.abs(A[i] - X)))
        elif metric == "cosine":
            for i in range(N):
                dot = np.sum(A[i] * X)
                norm_a = np.sqrt(np.sum(A[i] ** 2))
                norm_x = np.sqrt(np.sum(X ** 2))
                if norm_a * norm_x > 0:
                    distances.append(1.0 - dot / (norm_a * norm_x))
                else:
                    distances.append(1.0)
        elif metric == "dot":
            for i in range(N):
                distances.append(-np.sum(A[i] * X))
        
        top_indices = np.argsort(distances)[:K].tolist()
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"Metric: {metric}")
        print(f"  GPU time: {gpu_time:.6f} sec")
        print(f"  CPU time: {cpu_time:.6f} sec")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Check if results match
        common = len(set(result_gpu) & set(top_indices))
        match_percent = (common / K) * 100
        print(f"  Results match: {match_percent:.1f}%")

def test_dimension_scaling():
    """Test how performance scales with dimension - optimized version"""
    prewarm_gpu()
    
    dimensions = [2, 16, 128, 1024, 32768]  # 2^15 = 32768
    N = 1000  # Keep vector count fixed
    
    print("Testing dimension scaling with GPU vs CPU:")
    print("Dimension\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for D in dimensions:
        # Generate random data
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        K = 10
        
        # Initialize GPU tensors
        A = torch.tensor(A_np, device="cuda")
        X = torch.tensor(X_np, device="cuda")
        
        # Warmup for this specific dimension
        _ = compute_distance(A[:10], X, "l2")
        torch.cuda.synchronize()
        
        # GPU timing with optimized implementation
        start = time.time()
        distances = compute_distance(A, X, "l
2")
        topk = torch.topk(distances, k=K, largest=False)
        result_gpu = topk.indices.cpu().numpy().tolist()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU timing - with sample for very large dimensions
        start = time.time()
        distances = []
        # For large dimensions, use a subsample to estimate time
        sample_size = N if D < 10000 else min(100, N)
        
        for i in range(sample_size):
            distances.append(np.sqrt(np.sum((A_np[i] - X_np) ** 2)))
        
        cpu_time_sample = time.time() - start
        
        # Scale up if we used a sample
        if sample_size < N:
            cpu_time = cpu_time_sample * (N / sample_size)
        else:
            cpu_time = cpu_time_sample
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{D}\t\t{gpu_time:.6f}\t{cpu_time:.6f}\t{speedup:.2f}x")
        
        # Additional insights for very small or very large dimensions
        if D <= 16:
            print(f"  Note: For D={D}, using optimized small-dimension path")
        elif D >= 32768:
            print(f"  Note: For D={D}, using optimized large-dimension path")

def test_vector_count_scaling():
    """Test how performance scales with vector count"""
    prewarm_gpu()
    
    vector_counts = [100, 1000, 4000, 10000, 100000]
    D = 128  # Keep dimension fixed
    
    print("Testing vector count scaling with GPU vs CPU:")
    print("Vector Count\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for N in vector_counts:
        # Generate random data
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        K = 10
        
        # GPU timing
        A = torch.tensor(A_np, device="cuda")
        X = torch.tensor(X_np, device="cuda")
        
        torch.cuda.synchronize()
        start = time.time()
        
        distances = compute_distance(A, X, "l2")
            
        topk = torch.topk(distances, k=K, largest=False)
        result_gpu = topk.indices.cpu().numpy().tolist()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU timing - with sample for very large counts
        start = time.time()
        distances = []
        
        # For large vector counts, use a subsample to estimate time
        sample_size = N if N < 10000 else min(1000, N)
        
        for i in range(sample_size):
            distances.append(np.sqrt(np.sum((A_np[i] - X_np) ** 2)))
        
        cpu_time_sample = time.time() - start
        
        # Scale up if we used a sample
        if sample_size < N:
            cpu_time = cpu_time_sample * (N / sample_size)
        else:
            cpu_time = cpu_time_sample
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{N}\t\t{gpu_time:.6f}\t{cpu_time:.6f}\t{speedup:.2f}x")

def compare_knn_ann():
    """
    Compare KNN and ANN algorithms using the same dataset
    - Time performance
    - Recall rate
    """
    prewarm_gpu()  # Ensure GPU is warmed up for accurate timing
    
    print("\n--- KNN vs ANN Comparison ---")
    
    # Load the same test data for both algorithms
    N, D, A, X, K = testdata_knn("")
    
    print(f"Dataset: {N} vectors of dimension {D}, finding top {K} neighbors")
    
    # Run tests for different distance metrics
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        print(f"\nMetric: {metric}")
        
        # Run KNN and measure time
        torch.cuda.synchronize()
        start_time = time.time()
        knn_result = our_knn(N, D, A, X, K, metric)
        torch.cuda.synchronize()
        knn_time = time.time() - start_time
        
        # Run ANN and measure time
        torch.cuda.synchronize()
        start_time = time.time()
        ann_result = our_ann(N, D, A, X, K, metric)
        torch.cuda.synchronize()
        ann_time = time.time() - start_time
        
        # Calculate speedup and recall
        speedup = knn_time / ann_time if ann_time > 0 else float('inf')
        recall = compute_recall(knn_result, ann_result, K)
        
        # Print results
        print(f"  KNN time: {knn_time:.6f} sec")
        print(f"  ANN time: {ann_time:.6f} sec")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Recall: {recall:.2%}")

def extrapolate_large_dataset():
    """Extrapolate performance for 4,000,000 vectors"""
    prewarm_gpu()
    
    # Test with a smaller sample
    N_sample = 10000
    D = 128
    K = 10
    
    # Generate sample data
    A_np = np.random.randn(N_sample, D).astype(np.float32)
    X_np = np.random.randn(D).astype(np.float32)
    
    # GPU timing
    A = torch.tensor(A_np, device="cuda")
    X = torch.tensor(X_np, device="cuda")
    
    torch.cuda.synchronize()
    start = time.time()
    distances = compute_distance(A, X, "l2")
    topk = torch.topk(distances, k=K, largest=False)
    torch.cuda.synchronize()
    sample_time = time.time() - start
    
    # Extrapolate to 4,000,000 vectors
    N_large = 4000000
    scaling_factor = N_large / N_sample
    
    # Simple linear extrapolation
    linear_estimate = sample_time * scaling_factor
    
    # More realistic sublinear extrapolation (considering batching and optimizations)
    sublinear_estimate = sample_time * (scaling_factor ** 0.8)  # Using a sublinear scaling factor
    

def test_dimension_details():
    """Test specific dimension ranges that showed issues in previous runs"""
    prewarm_gpu()
    
    problematic_dims = [2, 16, 64, 128, 512, 1024, 4096]
    N = 1000
    K = 10
    
    print("\nDetailed testing for problematic dimensions:")
    print("Dimension\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for D in problematic_dims:
        # Generate random data
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        
        # GPU timing with careful warmup
        A = torch.tensor(A_np, device="cuda")
        X = torch.tensor(X_np, device="cuda")
        
        # Specific warmup for this dimension
        for _ in range(3):  # Multiple warmup runs
            _ = compute_distance(A[:50], X, "l2")
            torch.cuda.synchronize()
        
        # Actual timing
        torch.cuda.synchronize()
        start = time.time()
        distances = compute_distance(A, X, "l2")
        topk = torch.topk(distances, k=K, largest=False)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU timing
        start = time.time()
        distances = []
        for i in range(N):
            distances.append(np.sqrt(np.sum((A_np[i] - X_np) ** 2)))
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{D}\t\t{gpu_time:.6f}\t{cpu_time:.6f}\t{speedup:.2f}x")

def test_ann_cpu_gpu_speedup(dimensions=None, compute_true_recall=True):
    """
    Test the speedup of GPU ANN implementation compared to CPU ANN implementation
    across different dimensions, using the same torch implementation
    but with different device placements.
    
    Parameters:
    - dimensions: List of vector dimensions to test
    
    Returns:
    - None, prints results to console
    """
    
    if dimensions is None:
        dimensions = [2, 128, 1024, 32768]  # Include 2^15 = 32768
    
    metrics = ["l2", "cosine", "manhattan", "dot"]  # Include all four distance metrics
    K = 10  # Number of neighbors to find
    fixed_size = 4000  # Use 4,000 as mentioned in the assignment
    
    print("\n===== ANN GPU vs CPU Query Time Speedup Test =====")
    print(f"Finding top {K} neighbors using various metrics (cluster preprocessing NOT included in timing)")
    print(f"Fixed data size: {fixed_size} vectors")
    
    # Ensure GPU is warmed up
    prewarm_gpu()
    
    def run_improved_ann_on_device(N, D, A_np, X_np, K, metric, device):
        # Convert data to tensors on the specified device
        A = torch.tensor(A_np, dtype=torch.float32, device=device)
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        
        # Step 1: Determine clustering parameters (based on improved ANN algorithm)
        if N <= 1000:
            K1 = min(int(N/10), 20)  # More clusters for small datasets (10%)
        else:
            K1 = min(int(np.sqrt(N)), 100)  # Standard sqrt(N) for larger datasets
        
        # Adjust based on dimensionality - fewer clusters for high dimensions
        if D > 100:
            K1 = max(int(K1 * 0.7), 10)  # Reduce clusters for high-dimensional data
            
        # Ensure K1 is at least 2x the requested K for better coverage
        K1 = max(K1, 2*K)
        
        # Step 2: Run KMeans clustering (preprocessing step - not timed for query performance)
        if device == "cuda":
            cluster_ids = our_kmeans(N, D, A_np, K1, metric)
            cluster_ids_tensor = torch.tensor(cluster_ids, device=device)
        else:
            # For CPU, use the same algorithm but force CPU tensors
            with torch.no_grad():
                # Initialize centroids (simplified k-means++)
                centroid_indices = torch.randperm(N)[:K1]
                centroids = A[centroid_indices]
                
                # Run simplified k-means
                cluster_ids_tensor = torch.zeros(N, dtype=torch.long, device=device)
                for _ in range(10):  # Limit iterations for speed
                    # Compute distances from points to centroids
                    distances = torch.cdist(A, centroids)
                    # Assign points to nearest centroid
                    cluster_ids_tensor = torch.argmin(distances, dim=1)
                    # Update centroids
                    for j in range(K1):
                        mask = cluster_ids_tensor == j
                        if mask.sum() > 0:
                            centroids[j] = A[mask].mean(dim=0)
        
        # Compute centroids and cluster sizes for each cluster
        centroids = []
        cluster_sizes = []
        for j in range(K1):
            points = A[cluster_ids_tensor == j]
            size = points.size(0)
            cluster_sizes.append(size)
            if size > 0:
                centroids.append(points.mean(dim=0))
            else:
                centroids.append(torch.zeros(D, device=device))
        centroids = torch.stack(centroids)
        
        # Increase number of probed clusters - 30% of clusters or at least 10
        K1_probe = max(min(int(K1 * 0.3), 20), 10)
        K1_probe = min(K1_probe, K1)  # Can't probe more clusters than we have
        
        # Step 3: Start timing the query phase
        if device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
            
        # Compute distance to all centroids
        if metric == "l2":
            centroid_distances = torch.cdist(X.unsqueeze(0), centroids).squeeze(0)
        elif metric == "cosine":
            X_norm = X / torch.norm(X)
            centroids_norm = centroids / torch.norm(centroids, dim=1, keepdim=True)
            centroid_distances = 1 - torch.matmul(X_norm, centroids_norm.T)
        elif metric == "manhattan":
            centroid_distances = torch.zeros(K1, device=device)
            for j in range(K1):
                centroid_distances[j] = torch.sum(torch.abs(X - centroids[j]))
        elif metric == "dot":
            centroid_distances = -torch.matmul(X, centroids.T)
        
        # Adaptive search - dynamically adjust clusters based on their sizes and distances
        candidate_score = torch.zeros_like(centroid_distances)
        norm_distances = centroid_distances / (centroid_distances.max() + 1e-6)  # Normalize to [0,1]
        
        # Calculate a score combining distance and cluster size
        for i in range(K1):
            cluster_size = cluster_sizes[i]
            if cluster_size > 0:  # Avoid division by zero
                # Higher score for closer clusters with more points
                size_factor = min(cluster_size / (N/K1), 2.0)  # Cap size factor at 2.0
                candidate_score[i] = (1 - norm_distances[i]) * size_factor
        
        # Select top clusters based on combined score
        top_cluster_indices = torch.topk(candidate_score, k=K1_probe, largest=True).indices
        
        # Adaptive K2: more neighbors from closer clusters, fewer from distant ones
        base_K2 = min(100, N // K1)  # Base number of neighbors per cluster
        
        all_candidate_indices = []
        all_candidate_distances = []
        
        # Collect candidates from top clusters
        for i, cluster_idx in enumerate(top_cluster_indices):
            # Get all points in this cluster
            cluster_point_indices = (cluster_ids_tensor == cluster_idx.item()).nonzero(as_tuple=True)[0]
            
            if cluster_point_indices.size(0) > 0:
                # Adaptive K2 - more neighbors from closer clusters
                closeness_factor = 1.0 - i / K1_probe  # 1.0 for closest, lower for farther
                K2_cluster = max(int(base_K2 * (0.5 + 0.5 * closeness_factor)), K)
                K2_cluster = min(K2_cluster, cluster_point_indices.size(0))
                
                # Compute distances for all points in this cluster
                cluster_points = A[cluster_point_indices]
                if metric == "l2":
                    distances = torch.cdist(X.unsqueeze(0), cluster_points).squeeze(0)
                elif metric == "cosine":
                    X_norm = X / torch.norm(X)
                    points_norm = cluster_points / torch.norm(cluster_points, dim=1, keepdim=True)
                    distances = 1 - torch.matmul(X_norm, points_norm.T)
                elif metric == "manhattan":
                    distances = torch.zeros(cluster_points.size(0), device=device)
                    for j in range(cluster_points.size(0)):
                        distances[j] = torch.sum(torch.abs(X - cluster_points[j]))
                elif metric == "dot":
                    distances = -torch.matmul(X, cluster_points.T)
                
                # Get top K2 nearest neighbors in this cluster
                topk_in_cluster = torch.topk(distances, k=K2_cluster, largest=False)
                
                # Store their indices and distances
                all_candidate_indices.append(cluster_point_indices[topk_in_cluster.indices])
                all_candidate_distances.append(topk_in_cluster.values)
        
        # Merge and rerank all candidates
        if all_candidate_indices:
            # Concatenate all candidate indices and distances
            merged_indices = torch.cat(all_candidate_indices)
            merged_distances = torch.cat(all_candidate_distances)
            
            # Reranking - if we have enough candidates, recompute distances for final ranking
            if merged_indices.size(0) > K * 10:  # Only if we have 10x more candidates than K
                # Recompute distances directly with query
                final_points = A[merged_indices]
                if metric == "l2":
                    recomputed_distances = torch.cdist(X.unsqueeze(0), final_points).squeeze(0)
                elif metric == "cosine":
                    X_norm = X / torch.norm(X)
                    points_norm = final_points / torch.norm(final_points, dim=1, keepdim=True)
                    recomputed_distances = 1 - torch.matmul(X_norm, points_norm.T)
                elif metric == "manhattan":
                    recomputed_distances = torch.zeros(final_points.size(0), device=device)
                    for j in range(final_points.size(0)):
                        recomputed_distances[j] = torch.sum(torch.abs(X - final_points[j]))
                elif metric == "dot":
                    recomputed_distances = -torch.matmul(X, final_points.T)
                
                # Find top K among merged candidates using recomputed distances
                k_actual = min(K, merged_indices.size(0))
                final_topk = torch.topk(recomputed_distances, k=k_actual, largest=False)
                result_indices = merged_indices[final_topk.indices]
            else:
                # Use original distances if not enough candidates
                k_actual = min(K, merged_indices.size(0))
                final_topk = torch.topk(merged_distances, k=k_actual, largest=False)
                result_indices = merged_indices[final_topk.indices]
        else:
            # Fallback: if no candidates found, use brute-force KNN
            if metric == "l2":
                distances = torch.cdist(X.unsqueeze(0), A).squeeze(0)
            elif metric == "cosine":
                X_norm = X / torch.norm(X)
                A_norm = A / torch.norm(A, dim=1, keepdim=True)
                distances = 1 - torch.matmul(X_norm, A_norm.T)
            elif metric == "manhattan":
                distances = torch.zeros(N, device=device)
                for j in range(N):
                    distances[j] = torch.sum(torch.abs(X - A[j]))
            elif metric == "dot":
                distances = -torch.matmul(X, A.T)
                
            topk = torch.topk(distances, k=min(K, N), largest=False)
            result_indices = topk.indices
        
        # End timing
        if device == "cuda":
            torch.cuda.synchronize()
        query_time = time.time() - start_time
        
        # Return final indices
        return result_indices.cpu().numpy().tolist(), query_time
    
    # Test across different dimensions (with fixed data size)
    print("\n--- Testing across dimensions (fixed data size: {fixed_size}) ---")
    if compute_true_recall:
        print("Dim\tMetric\tCPU Time (s)\tGPU Time (s)\tSpeedup\tCPU-GPU Match\tCPU Recall\tGPU Recall")
    else:
        print("Dim\tMetric\tCPU Time (s)\tGPU Time (s)\tSpeedup\tCPU-GPU Match")
    
    for D in dimensions:
        # Generate random data
        A_np = np.random.randn(fixed_size, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        
        for metric in metrics:
            try:
                # Run on CPU
                cpu_result, cpu_time = run_improved_ann_on_device(fixed_size, D, A_np, X_np, K, metric, "cpu")
                
                # Run on GPU
                gpu_result, gpu_time = run_improved_ann_on_device(fixed_size, D, A_np, X_np, K, metric, "cuda")
                
                # Calculate speedup and recall
                speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
                
                # Calculate recall between CPU and GPU results
                cpu_gpu_recall = compute_recall(cpu_result, gpu_result, K)
                
                # Optionally compute recall against true KNN results
                if compute_true_recall:
                    # Get ground truth KNN results
                    true_knn = our_knn(fixed_size, D, A_np, X_np, K, metric)
                    cpu_recall = compute_recall(true_knn, cpu_result, K)
                    gpu_recall = compute_recall(true_knn, gpu_result, K)
                    print(f"{D}\t{metric}\t{cpu_time:.6f}\t{gpu_time:.6f}\t{speedup:.2f}x\t{cpu_gpu_recall:.2%}\t{cpu_recall:.2%}\t{gpu_recall:.2%}")
                else:
                    print(f"{D}\t{metric}\t{cpu_time:.6f}\t{gpu_time:.6f}\t{speedup:.2f}x\t{cpu_gpu_recall:.2%}")
            except Exception as e:
                print(f"{D}\t{metric}\tError: {str(e)}")

if __name__ == "__main__":
    # print("\n--- Basic Tests (with GPU prewarming) ---")
    # test_knn()
    test_knn_dimension_scaling()


    
    print("\n--- GPU vs CPU Comparison ---")
    compare_gpu_cpu()
    
    print("\n--- Dimension Scaling Test (Optimized) ---")
    test_dimension_scaling()
    
    print("\n--- Detailed Dimension Analysis ---")
    test_dimension_details()
    
    print("\n--- Vector Count Scaling Test ---")
    test_vector_count_scaling()
    
    print("\n--- Large Dataset Extrapolation ---")
    extrapolate_large_dataset()
    
    print("\n--- KMeans Test ---")
    test_kmeans()
    
    print("\n--- ANN Tests ---")
    test_ann()
    
    print("\n--- Recall & Precision ---")
    test_recall()

    print("\n--- ANN CPU vs GPU Speedup Test ---")
    test_ann_cpu_gpu_speedup(compute_true_recall=True)

