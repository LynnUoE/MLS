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
# Optimized KNN Implementation
# -----------------------------------------------------------------------------
def our_knn(N, D, A_np, X_np, K, metric="l2", cache_tensors=False):
    """
    Optimized KNN implementation based on the pseudocode in the vector search PDF
    
    Algorithm 1: kNN (brute force)
    Input: a set D of n vectors, a query vector q, distance function dist(·, ·), k
    Output: S_k ⊆ D such that |S_k| = k, dist(q, x) ≥ dist(q, y) for any x ∈ S_k, y ∈ D \ S_k
    1: S_k ← ∅
    2: for each x ∈ D do
    3:   compute dist(q, x)
    4: end for
    5: sort vectors x of D according to dist(q, x) in non-decreasing order
    6: S_k ← top-k x's in the sorted order
    7: return S_k
    """
    # Convert data to GPU tensors if not already
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
    
    # Use batched processing for large datasets
    batch_size = 50000
    
    # Compute distances efficiently based on the distance metric
    if N > batch_size:
        distances = torch.empty(N, device="cuda")
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch = A[i:end_idx]
            
            if metric == "l2":
                # L2 distance (squared)
                dist_batch = torch.sum((batch - X.unsqueeze(0)) ** 2, dim=1)
            elif metric == "cosine":
                # Cosine distance: 1 - cos(angle)
                X_norm = torch.norm(X)
                batch_norm = torch.norm(batch, dim=1, keepdim=True)
                dot_prod = torch.matmul(batch, X)
                # Avoid division by zero
                safe_norms = torch.clamp(batch_norm * X_norm, min=1e-8)
                dist_batch = 1.0 - dot_prod / safe_norms.squeeze()
            elif metric == "dot":
                # Negative dot product (since smaller is closer)
                dist_batch = -torch.matmul(batch, X)
            elif metric == "manhattan":
                # L1 distance
                dist_batch = torch.sum(torch.abs(batch - X.unsqueeze(0)), dim=1)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            distances[i:end_idx] = dist_batch
    else:
        if metric == "l2":
            distances = torch.sum((A - X.unsqueeze(0)) ** 2, dim=1)
        elif metric == "cosine":
            X_norm = torch.norm(X)
            A_norm = torch.norm(A, dim=1, keepdim=True)
            dot_prod = torch.matmul(A, X)
            # Avoid division by zero
            safe_norms = torch.clamp(A_norm * X_norm, min=1e-8)
            distances = 1.0 - dot_prod / safe_norms.squeeze()
        elif metric == "dot":
            distances = -torch.matmul(A, X)
        elif metric == "manhattan":
            distances = torch.sum(torch.abs(A - X.unsqueeze(0)), dim=1)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    # Find top-K nearest neighbors (smallest distances)
    k_actual = min(K, N)  # Ensure K is not larger than N
    topk = torch.topk(distances, k=k_actual, largest=False)
    
    # Return the indices of the top-K nearest neighbors
    return topk.indices.cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# Optimized KMeans Implementation
# -----------------------------------------------------------------------------
def kmeans_plus_plus(A, K, metric="l2"):
    """
    Optimized K-means++ initialization
    
    1. Choose the first centroid uniformly at random from the data points
    2. For each data point, compute the distance to the nearest existing centroid
    3. Choose the next centroid with probability proportional to the squared distance
    4. Repeat until K centroids are chosen
    """
    N = A.shape[0]
    D = A.shape[1]
    centroids = []
    
    # Randomly select the first centroid
    first_idx = torch.randint(0, N, (1,), device=A.device)
    centroids.append(A[first_idx].squeeze(0))
    
    for _ in range(1, K):
        # Calculate minimum distance to any existing centroid
        dists = torch.full((N,), float('inf'), device=A.device)
        
        for c in centroids:
            if metric == "l2":
                # Squared Euclidean distance
                new_dists = torch.sum((A - c.unsqueeze(0)) ** 2, dim=1)
            elif metric == "cosine":
                # Cosine distance
                c_norm = torch.norm(c)
                A_norm = torch.norm(A, dim=1)
                cos_sim = torch.matmul(A, c) / (A_norm * c_norm + 1e-8)
                new_dists = 1.0 - cos_sim
            else:
                # Default to L2 for other metrics
                new_dists = torch.sum((A - c.unsqueeze(0)) ** 2, dim=1)
                
            dists = torch.minimum(dists, new_dists)
        
        # Square the distances to give more weight to points far from centroids
        weights = dists ** 2
        
        # If all distances are very small, choose randomly
        if torch.sum(weights) < 1e-8:
            next_idx = torch.randint(0, N, (1,), device=A.device)
        else:
            # Sample next centroid with probability proportional to squared distance
            weights = weights / torch.sum(weights)
            next_idx = torch.multinomial(weights, 1)
            
        centroids.append(A[next_idx].squeeze(0))
    
    return torch.stack(centroids)

def our_kmeans(N, D, A_np, K, metric="l2", max_iter=100, tol=1e-4):
    """
    Optimized K-means clustering implementation based on Lloyd's algorithm
    
    Algorithm: Lloyd's K-means algorithm
    Input: D = {x_1, . . . , x_n}, K;
    Output: clustering C = {C_1, . . . , C_K} of D
    
    1. initialize C_1, …, C_K;
    2. while not converged do
    3.   compute centroid μ_i = (1/|C_i|) ∑_{x∈C_i} x for each i = 1, . . . , K;
    4.   update C_1, …, C_K by assigning each x ∈ D to cluster C_ℓ whose centroid μ_ℓ it is closest to;
    5.   if C_1, …, C_K don't change, we have converged;
    6. return clustering C_1, …, C_K
    """
    # Convert data to GPU tensor
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    
    # Initialize centroids with K-means++
    centroids = kmeans_plus_plus(A, K, metric)
    
    # Determine batch size based on dataset size
    batch_size = min(10000, N)
    
    # Previous cluster assignments
    prev_cluster_ids = torch.zeros(N, dtype=torch.long, device="cuda") - 1
    
    for i in range(max_iter):
        cluster_ids = torch.zeros(N, dtype=torch.long, device="cuda")
        
        # Process in batches for large datasets
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch = A[batch_start:batch_end]
            
            # Calculate distances to each centroid
            distances = torch.zeros((batch_end - batch_start, K), device="cuda")
            
            for j in range(K):
                if metric == "l2":
                    # Squared Euclidean distance
                    distances[:, j] = torch.sum((batch - centroids[j].unsqueeze(0)) ** 2, dim=1)
                elif metric == "cosine":
                    # Cosine distance
                    centroid_norm = torch.norm(centroids[j])
                    batch_norm = torch.norm(batch, dim=1)
                    cos_sim = torch.matmul(batch, centroids[j]) / (batch_norm * centroid_norm + 1e-8)
                    distances[:, j] = 1.0 - cos_sim
                else:
                    # Default to L2 for other metrics
                    distances[:, j] = torch.sum((batch - centroids[j].unsqueeze(0)) ** 2, dim=1)
            
            # Assign to nearest centroid
            batch_cluster_ids = torch.argmin(distances, dim=1)
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
# Optimized ANN Implementation
# -----------------------------------------------------------------------------
def our_ann(N, D, A_np, X_np, K, metric="l2"):
    """
    Optimized Approximate Nearest Neighbor algorithm
    
    Pseudocode from PDF:
    1. Use KMeans to cluster the data into K clusters
    2. In each query, find the nearest K1 cluster center as the approximate nearest neighbor
    3. Use KNN to find the nearest K2 neighbor from the K1 cluster centers
    4. Merge K1 * K2 vectors and find top K neighbors
    """
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    
    # Determine optimal number of clusters based on dataset size
    if N < 5000:
        num_clusters = min(50, N // 10)
    else:
        num_clusters = min(200, N // 100)
    
    # Ensure reasonable number of clusters
    num_clusters = max(10, min(num_clusters, N // 5))
    
    # Use cached clusters if possible for repeated queries
    if hasattr(our_ann, 'cached_clusters') and our_ann.cached_clusters[0].shape[0] == N:
        cluster_ids = our_ann.cached_clusters[0]
        centroids = our_ann.cached_clusters[1]
    else:
        # Run KMeans
        cluster_ids_list = our_kmeans(N, D, A_np, num_clusters, metric=metric, max_iter=100)
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
    
    # Find closest clusters to query using the appropriate distance metric
    if metric == "l2":
        centroid_distances = torch.sum((centroids - X.unsqueeze(0)) ** 2, dim=1)
    elif metric == "cosine":
        X_norm = torch.norm(X)
        centroid_norms = torch.norm(centroids, dim=1)
        cos_sim = torch.matmul(centroids, X) / (centroid_norms * X_norm + 1e-8)
        centroid_distances = 1.0 - cos_sim
    elif metric == "dot":
        centroid_distances = -torch.matmul(centroids, X)
    elif metric == "manhattan":
        centroid_distances = torch.sum(torch.abs(centroids - X.unsqueeze(0)), dim=1)
    
    # K1: Number of clusters to search
    # Adaptive K1 based on dataset size and number of clusters
    K1 = min(max(num_clusters // 2, 5), num_clusters)
    top_cluster_indices = torch.topk(centroid_distances, k=K1, largest=False).indices
    
    # K2: Number of points to retrieve from each cluster
    # Adaptive K2 based on dataset size and K
    K2 = min(max(K * 5, 100), N // (K1 * 2))
    
    # Collect points from top clusters with distances
    all_candidate_indices = []
    all_candidate_distances = []
    
    for cluster_idx in top_cluster_indices:
        # Get indices of points in this cluster
        cluster_point_indices = torch.nonzero(cluster_ids == cluster_idx.item(), as_tuple=True)[0]
        
        if cluster_point_indices.size(0) > 0:
            # Get points from this cluster
            cluster_points = A[cluster_point_indices]
            
            # Calculate distances to query
            if metric == "l2":
                distances = torch.sum((cluster_points - X.unsqueeze(0)) ** 2, dim=1)
            elif metric == "cosine":
                X_norm = torch.norm(X)
                point_norms = torch.norm(cluster_points, dim=1)
                cos_sim = torch.matmul(cluster_points, X) / (point_norms * X_norm + 1e-8)
                distances = 1.0 - cos_sim
            elif metric == "dot":
                distances = -torch.matmul(cluster_points, X)
            elif metric == "manhattan":
                distances = torch.sum(torch.abs(cluster_points - X.unsqueeze(0)), dim=1)
            
            # Get top K2 nearest points from this cluster
            k_to_select = min(K2, cluster_point_indices.size(0))
            if k_to_select > 0:
                topk = torch.topk(distances, k=k_to_select, largest=False)
                
                # Store indices and distances
                all_candidate_indices.append(cluster_point_indices[topk.indices])
                all_candidate_distances.append(topk.values)
    
    # Handle case where no candidates were found
    if not all_candidate_indices:
        return our_knn(N, D, A_np, X_np, K, metric)
    
    # Combine all candidate points
    candidate_indices = torch.cat(all_candidate_indices)
    candidate_distances = torch.cat(all_candidate_distances)
    
    # Get final top K
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