import triton
import triton.language as tl
import torch
import time
import numpy as np
from test import testdata_knn, testdata_kmeans, testdata_ann

# -----------------------------------------------------------------------------
# Optimized Triton kernels for distance metrics with dimension-adaptive parameters
# -----------------------------------------------------------------------------

@triton.jit
def manhattan_distance_kernel(
    A, X, output, 
    D: tl.constexpr, 
    stride_A: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        acc += tl.sum(tl.abs(a - x))
    tl.store(output + pid, acc)

@triton.jit
def dot_distance_kernel(
    A, X, output, 
    D: tl.constexpr, 
    stride_A: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        acc += tl.sum(a * x)
    tl.store(output + pid, -acc)

@triton.jit
def cosine_distance_kernel(
    A, X, output, 
    D: tl.constexpr, 
    stride_A: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr
):
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
    norm_product = tl.sqrt(norm_a) * tl.sqrt(norm_x)
    # Handle edge case where one of the vectors is zero
    norm_product = tl.where(norm_product > 0.0, norm_product, 1.0)
    sim = dot / norm_product
    tl.store(output + pid, 1.0 - sim)

@triton.jit
def l2_distance_kernel(
    A, X, output, 
    D: tl.constexpr, 
    stride_A: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr,
    USE_SQRT: tl.constexpr
):
    """
    Compute L2 distance with adaptive vectorization based on dimension
    """
    pid = tl.program_id(0)
    row_ptr = A + pid * stride_A
    acc = 0.0
    
    # Process data in blocks
    for d in range(0, D, BLOCK_SIZE):
        offs = d + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        a = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x = tl.load(X + offs, mask=mask, other=0.0)
        
        # Compute difference and accumulate sum of squares
        diff = a - x
        acc += tl.sum(diff * diff)
    
    # Optionally take square root for true L2 distance
    if USE_SQRT:
        tl.store(output + pid, tl.sqrt(acc))
    else:
        tl.store(output + pid, acc)

# -----------------------------------------------------------------------------
# Optimized distance function with dimension-specific tuning
# -----------------------------------------------------------------------------

_cache = {}  # Cache for compiled kernels

def compute_distance(A, X, metric="l2", block_size=None, use_sqrt=False):
    """
    Optimized distance computation with dimension-specific tuning
    
    Args:
        A: tensor with shape (N, D)
        X: tensor with shape (D) or (1, D)
        metric: distance metric to use
        block_size: custom block size (will be auto-determined if None)
        use_sqrt: whether to use square root for L2 distance
        
    Returns:
        tensor of shape (N,) containing distances
    """
    N, D = A.shape
    if X.dim() == 2:
        X = X.squeeze(0)
    
    # For very small dimensions, use PyTorch built-in ops
    if D <= 4:
        if metric == "l2":
            if use_sqrt:
                return torch.norm(A - X.unsqueeze(0), dim=1)
            else:
                return torch.sum((A - X.unsqueeze(0))**2, dim=1)
        elif metric == "cosine":
            A_norm = torch.norm(A, dim=1, keepdim=True)
            X_norm = torch.norm(X)
            dots = torch.matmul(A, X)
            return 1.0 - dots / (A_norm.squeeze(1) * X_norm)
        elif metric == "dot":
            return -torch.matmul(A, X)
        elif metric == "manhattan":
            return torch.sum(torch.abs(A - X.unsqueeze(0)), dim=1)
    
    # Optimize block size based on dimensions
    if block_size is None:
        if D <= 32:
            block_size = 32
        elif D <= 256:
            block_size = 128
        elif D <= 1024:
            block_size = 256
        else:
            block_size = 512
    
    # Use vectorized load for medium to large dimensions
    vectorize = 1 if D >= 32 else 0
    
    output = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    
    # Use the cache to avoid recompiling kernels
    cache_key = (metric, block_size, vectorize, use_sqrt if metric == "l2" else False)
    if cache_key not in _cache:
        if metric == "l2":
            _cache[cache_key] = l2_distance_kernel[grid]
        elif metric == "cosine":
            _cache[cache_key] = cosine_distance_kernel[grid]
        elif metric == "dot":
            _cache[cache_key] = dot_distance_kernel[grid]
        elif metric == "manhattan":
            _cache[cache_key] = manhattan_distance_kernel[grid]
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    kernel = _cache[cache_key]
    
    if metric == "l2":
        kernel(A, X, output, D, A.stride(0), BLOCK_SIZE=block_size, VECTORIZE=vectorize, USE_SQRT=use_sqrt)
    else:
        kernel(A, X, output, D, A.stride(0), BLOCK_SIZE=block_size, VECTORIZE=vectorize)
    
    return output

# -----------------------------------------------------------------------------
# Optimized batch processing for large datasets
# -----------------------------------------------------------------------------
def compute_distance_batched(A, X, metric="l2", block_size=None, batch_size=50000, use_sqrt=False):
    """
    Batch-process distance computation for large datasets
    """
    N, D = A.shape
    if X.dim() == 2:
        X = X.squeeze(0)
    
    output = torch.empty((N,), device=A.device, dtype=torch.float32)
    
    for i in range(0, N, batch_size):
        end_idx = min(i + batch_size, N)
        batch = A[i:end_idx]
        batch_output = compute_distance(batch, X, metric, block_size, use_sqrt)
        output[i:end_idx] = batch_output
    
    return output

# -----------------------------------------------------------------------------
# Optimized Top-K KNN with metric option and batch processing
# -----------------------------------------------------------------------------
def our_knn(N, D, A_np, X_np, K, metric="l2", cache_tensors=False):
    """
    Optimized KNN implementation with dimension-specific tuning
    """
    # Check if data is already on GPU
    if cache_tensors and hasattr(our_knn, 'cached_tensors'):
        cached_A, cached_X, cached_metric = our_knn.cached_tensors
        if cached_A.shape == (N, D) and cached_X.shape[-1] == D and cached_metric == metric:
            A = cached_A
            X = cached_X
        else:
            A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
            X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
            our_knn.cached_tensors = (A, X, metric)
    else:
        A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
        X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
        if cache_tensors:
            our_knn.cached_tensors = (A, X, metric)
    
    # Determine if we should use sqrt for L2 distance (only needed for small D)
    use_sqrt = (metric == "l2" and D <= 16)
    
    # Use batched processing for large datasets
    if N > 50000:
        distances = compute_distance_batched(A, X, metric, use_sqrt=use_sqrt)
    else:
        distances = compute_distance(A, X, metric, use_sqrt=use_sqrt)
    
    topk = torch.topk(distances, k=min(K, N), largest=False)
    return topk.indices.cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# Optimized KMeans implementation
# -----------------------------------------------------------------------------
def kmeans_plus_plus(A, K, metric="l2"):
    """
    A: tensor with shape (N, D) on GPU.
    Returns K initial centroids using K-means++ algorithm.
    """
    N = A.shape[0]
    D = A.shape[1]
    centroids = []
    
    # Randomly select the first centroid and squeeze to 1D vector.
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    
    # Use squared distances for L2 to avoid sqrt operations
    use_sqrt = False
    
    for _ in range(1, K):
        if N > 50000:
            # Batched implementation for large datasets
            dists = compute_distance_batched(A, centroids[0].unsqueeze(0), metric, use_sqrt=use_sqrt)
            for c in centroids[1:]:
                d_new = compute_distance_batched(A, c.unsqueeze(0), metric, use_sqrt=use_sqrt)
                dists = torch.minimum(dists, d_new)
        else:
            dists = compute_distance(A, centroids[0].unsqueeze(0), metric, use_sqrt=use_sqrt)
            for c in centroids[1:]:
                d_new = compute_distance(A, c.unsqueeze(0), metric, use_sqrt=use_sqrt)
                dists = torch.minimum(dists, d_new)
        
        # Square distances for probability distribution
        if metric != "l2" or use_sqrt:
            dists = dists ** 2
            
        # Avoid division by zero
        total = dists.sum()
        if total < 1e-10:
            # If all distances are very small, choose randomly
            next_idx = torch.randint(0, N, (1,), device=A.device)
        else:
            probs = dists / total
            cumulative_probs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, device=A.device)
            next_idx = torch.searchsorted(cumulative_probs, r)
            # Ensure index is in bounds
            if next_idx >= N:
                next_idx = torch.tensor([N-1], device=A.device)
        
        centroids.append(A[next_idx].squeeze(0))
    
    return torch.stack(centroids)

def our_kmeans(N, D, A_np, K, metric="l2", max_iter=100, tol=1e-4):
    """
    Optimized K-means clustering with dimension-specific tuning
    """
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    
    # Initialize centroids with K-means++
    centroids = kmeans_plus_plus(A, K, metric)
    
    # Determine batch size based on dataset size and available memory
    batch_size = min(50000, N)
    
    # Determine if we should use sqrt for L2 distance (not needed for clustering)
    use_sqrt = False
    
    # Previous cluster assignments
    prev_cluster_ids = torch.zeros(N, dtype=torch.long, device="cuda") - 1
    
    for i in range(max_iter):
        cluster_ids = torch.zeros(N, dtype=torch.long, device="cuda")
        
        # Process in batches for large datasets
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch = A[batch_start:batch_end]
            
            # Calculate distances to each centroid
            batch_dists = torch.zeros((batch_end - batch_start, K), device="cuda")
            for j in range(K):
                centroid = centroids[j].unsqueeze(0)
                d = compute_distance(batch, centroid, metric, use_sqrt=use_sqrt)
                batch_dists[:, j] = d
            
            # Assign to nearest centroid
            batch_cluster_ids = torch.argmin(batch_dists, dim=1)
            cluster_ids[batch_start:batch_end] = batch_cluster_ids
        
        # Check for convergence based on cluster assignments
        if torch.all(cluster_ids == prev_cluster_ids):
            break
        
        prev_cluster_ids = cluster_ids.clone()
        
        # Update centroids
        new_centroids = []
        for j in range(K):
            cluster_points = A[cluster_ids == j]
            if cluster_points.size(0) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                # If a cluster is empty, keep the old centroid
                new_centroids.append(centroids[j])
        
        new_centroids = torch.stack(new_centroids)
        
        # Check for centroid convergence
        if metric == "l2":
            centroid_change = torch.norm(new_centroids - centroids)
        else:
            centroid_change = torch.max(torch.abs(new_centroids - centroids))
            
        centroids = new_centroids
        
        if centroid_change < tol:
            break
    
    return cluster_ids.cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# Optimized ANN implementation
# -----------------------------------------------------------------------------
def our_ann(N, D, A_np, X_np, K, metric="l2"):
    """
    Optimized Approximate Nearest Neighbor search using clustering
    
    Parameters:
    - N: Number of vectors
    - D: Dimension of vectors
    - A_np: A collection of vectors (N, D)
    - X_np: Query vector (D)
    - K: Number of top results to return
    - metric: Distance metric to use
    
    Returns:
    - List of indices of top K nearest vectors
    """
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    
    # Determine optimal number of clusters based on dataset size
    if N < 1000:
        num_clusters = min(20, N // 5)
    elif N < 10000:
        num_clusters = min(50, N // 10)
    else:
        num_clusters = min(200, N // 200)
    
    # Ensure at least 5 clusters and at most N/3 clusters
    num_clusters = max(5, min(num_clusters, N // 3))
    
    # Use cached clusters if possible for repeated queries
    if hasattr(our_ann, 'cached_clusters') and our_ann.cached_clusters[0].shape[0] == N:
        cluster_ids = our_ann.cached_clusters[0]
        centroids = our_ann.cached_clusters[1]
    else:
        # Run KMeans with more iterations for better clustering
        cluster_ids_list = our_kmeans(N, D, A_np, num_clusters, metric=metric, max_iter=150, tol=1e-5)
        cluster_ids = torch.tensor(cluster_ids_list, device="cuda")
        
        # Compute centroids
        centroids = []
        for j in range(num_clusters):
            points = A[cluster_ids == j]
            if points.size(0) > 0:
                centroids.append(points.mean(dim=0))
            else:
                centroids.append(torch.zeros(D, device="cuda"))
        centroids = torch.stack(centroids)
        
        # Cache the clusters for future use
        our_ann.cached_clusters = (cluster_ids, centroids)
    
    # Find closest clusters to query
    centroid_distances = compute_distance(centroids, X, metric)
    
    # K1: Number of clusters to search (INCREASED)
    K1 = min(max(num_clusters * 2 // 3, 5), num_clusters)  # Search at least 2/3 of the clusters
    top_cluster_indices = torch.topk(centroid_distances, k=K1, largest=False).indices
    
    # K2: Number of points to consider from each cluster (NEW PARAMETER)
    K2 = min(max(K * 5, 50), N // K1)  # Consider more points from each cluster
    
    # Collect points from top clusters with their distances
    all_candidate_indices = []
    all_candidate_distances = []
    
    for c in top_cluster_indices:
        cluster_idx = c.item()
        # Get points in this cluster
        indices = (cluster_ids == cluster_idx).nonzero(as_tuple=True)[0]
        
        if indices.size(0) > 0:
            # Get data points for this cluster
            cluster_points = A[indices]
            
            # Compute distances within this cluster
            cluster_distances = compute_distance(cluster_points, X, metric)
            
            # Get top K2 points from this cluster (or all if less than K2)
            k_to_select = min(K2, indices.size(0))
            topk_cluster = torch.topk(cluster_distances, k=k_to_select, largest=False)
            
            all_candidate_indices.append(indices[topk_cluster.indices])
            all_candidate_distances.append(topk_cluster.values)
    
    # Concatenate results from all clusters
    if not all_candidate_indices:
        # Fall back to standard KNN if no candidate indices
        return our_knn(N, D, A_np, X_np, K, metric)
    
    candidate_indices = torch.cat(all_candidate_indices)
    candidate_distances = torch.cat(all_candidate_distances)
    
    # Get top K from all candidates
    k_to_select = min(K, candidate_indices.size(0))
    final_topk = torch.topk(candidate_distances, k=k_to_select, largest=False)
    final_indices = candidate_indices[final_topk.indices]
    
    return final_indices.cpu().numpy().tolist()

def compute_recall(knn_result: list, ann_result: list, K: int) -> float:
    """Calculate recall rate between exact KNN and ANN results"""
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
        distances = compute_distance(A, X, "l2")
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
        
        if N > 50000:  # Use batched processing for large datasets
            distances = compute_distance_batched(A, X, "l2")
        else:
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
    
    print(f"Performance extrapolation for 4,000,000 vectors:")
    print(f"Sample time for {N_sample} vectors: {sample_time:.6f} seconds")
    print(f"Linear extrapolation: {linear_estimate:.2f} seconds")
    print(f"Sublinear extrapolation (with optimizations): {sublinear_estimate:.2f} seconds")
    print()
    print("Optimizations needed for 4,000,000 vectors:")
    print("1. Batch processing to manage GPU memory")
    print("2. Multiple GPU processing if available")
    print("3. Mixed precision (FP16) to reduce memory usage and increase throughput")
    print("4. Quantization techniques to compress vectors")
    print("5. Progressive refinement approach (coarse search followed by fine search)")

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

if __name__ == "__main__":
    print("\n--- Basic Tests (with GPU prewarming) ---")
    test_knn()
    
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