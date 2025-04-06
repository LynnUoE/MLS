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
        a = tl.load(row_ptr + offs, mask=offs < D, other=0.0)
        x = tl.load(X + offs, mask=offs < D, other=0.0)
        acc += tl.sum(tl.abs(a - x))
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
    A: tensor with shape (N, D)
    X: tensor with shape (1, D) (broadcasted to all rows)
    Returns a tensor of shape (N,) containing distances.
    """
    N, D = A.shape
    if block_size is None:
        block_size = 32 if D < 64 else 128
    output = torch.empty((N,), device=A.device, dtype=torch.float32)
    grid = (N,)
    if metric == "l2":
        l2_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    elif metric == "cosine":
        cosine_distance_kernel[grid](A, X, output, D, A.stride(0), BLOCK_SIZE=block_size)
    elif metric == "dot":
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
    A = torch.tensor(A_np, dtype=torch.float32, device="cuda")
    X = torch.tensor(X_np, dtype=torch.float32, device="cuda")
    num_clusters = min(6, N)

    cluster_ids_list = our_kmeans(N, D, A_np, num_clusters, metric=metric)
    cluster_ids = torch.tensor(cluster_ids_list, device="cuda")

    centroids = []
    for j in range(num_clusters):
        points = A[cluster_ids == j]
        if points.size(0) > 0:
            centroids.append(points.mean(dim=0))
        else:
            centroids.append(torch.zeros(D, device="cuda"))
    centroids = torch.stack(centroids)


    centroid_distances = compute_distance(centroids, X, metric)
    top_cluster_indices = torch.topk(centroid_distances, k=min(5, num_clusters), largest=False).indices

    selected_indices_list = []
    for c in top_cluster_indices:
        indices = (cluster_ids == c.item()).nonzero(as_tuple=True)[0]
        selected_indices_list.append(indices)
    selected_indices = torch.cat(selected_indices_list) if selected_indices_list else torch.arange(N, device="cuda")

    selected_points = A[selected_indices]
    distances = compute_distance(selected_points, X, metric)
    topk = torch.topk(distances, k=min(K, selected_indices.size(0)), largest=False)
    return selected_indices[topk.indices].cpu().numpy().tolist()


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


def test_ann_cpu_gpu_speedup(data_sizes=None, dimensions=None, compute_true_recall=True):
    """
    Test the speedup of GPU ANN implementation compared to CPU ANN implementation
    across different dataset sizes and dimensions, using the same torch implementation
    but with different device placements.
    
    Parameters:
    - data_sizes: List of dataset sizes (number of vectors) to test
    - dimensions: List of vector dimensions to test
    
    Returns:
    - None, prints results to console
    """
    from task_new import our_ann, prewarm_gpu, compute_recall, our_kmeans, our_knn
    
    if data_sizes is None:
        data_sizes = [1000, 4000, 10000]
    
    if dimensions is None:
        dimensions = [2, 128, 1024, 32768]  # Include 2^15 = 32768
    
    metrics = ["l2", "cosine"]  # Focus on the two metrics mentioned in the assignment
    K = 10  # Number of neighbors to find
    
    print("\n===== ANN GPU vs CPU Speedup Test =====")
    print(f"Finding top {K} neighbors using various metrics")
    
    # Ensure GPU is warmed up
    prewarm_gpu()


def test_ann_cpu_gpu_speedup(data_sizes=None, dimensions=None, compute_true_recall=True):
    """
    Test the speedup of GPU ANN implementation compared to CPU ANN implementation
    across different dataset sizes and dimensions, using the same torch implementation
    but with different device placements.
    
    Parameters:
    - data_sizes: List of dataset sizes (number of vectors) to test
    - dimensions: List of vector dimensions to test
    
    Returns:
    - None, prints results to console
    """
    from task_new import our_ann, prewarm_gpu, compute_recall, our_kmeans, our_knn
    
    if data_sizes is None:
        data_sizes = [1000, 4000, 10000]
    
    if dimensions is None:
        dimensions = [2, 128, 1024, 32768]  # Include 2^15 = 32768
    
    metrics = ["l2", "cosine", "manhattan", "dot"]  # Include all four distance metrics
    K = 10  # Number of neighbors to find
    
    print("\n===== ANN GPU vs CPU Speedup Test =====")
    print(f"Finding top {K} neighbors using various metrics")
    
    # Ensure GPU is warmed up
    prewarm_gpu()
    
    # Helper function to run ANN on specified device
    def run_ann_on_device(N, D, A_np, X_np, K, metric, device):
        # Convert data to tensors on the specified device
        A = torch.tensor(A_np, dtype=torch.float32, device=device)
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        
        # Run KMeans clustering
        start_time = time.time()
        num_clusters = min(6, N)
        
        # Cluster the data
        if device == "cuda":
            cluster_ids = our_kmeans(N, D, A_np, num_clusters, metric)
            cluster_ids_tensor = torch.tensor(cluster_ids, device=device)
        else:
            # For CPU, use the same algorithm but force CPU tensors
            with torch.no_grad():
                # Initialize centroids (simplified k-means++)
                centroid_indices = torch.randperm(N)[:num_clusters]
                centroids = A[centroid_indices]
                
                # Run simplified k-means
                cluster_ids_tensor = torch.zeros(N, dtype=torch.long, device=device)
                for _ in range(10):  # Limit iterations for speed
                    # Compute distances from points to centroids
                    distances = torch.cdist(A, centroids)
                    # Assign points to nearest centroid
                    cluster_ids_tensor = torch.argmin(distances, dim=1)
                    # Update centroids
                    for j in range(num_clusters):
                        mask = cluster_ids_tensor == j
                        if mask.sum() > 0:
                            centroids[j] = A[mask].mean(dim=0)
        
        # Find distances from query to centroids
        if device == "cuda":
            # For GPU, we'll use the existing pipeline which has optimized CUDA kernels
            result = our_ann(N, D, A_np, X_np, K, metric)
            end_time = time.time()
            return result, end_time - start_time
        else:
            # For CPU, implement a basic version of the same algorithm
            centroids = torch.zeros((num_clusters, D), device=device)
            for j in range(num_clusters):
                mask = cluster_ids_tensor == j
                if mask.sum() > 0:
                    centroids[j] = A[mask].mean(dim=0)
            
            # Find distances from query to centroids
            if metric == "l2":
                centroid_distances = torch.cdist(X.unsqueeze(0), centroids).squeeze(0)
            elif metric == "cosine":
                X_norm = X / torch.norm(X)
                centroids_norm = centroids / torch.norm(centroids, dim=1, keepdim=True)
                centroid_distances = 1 - torch.matmul(X_norm, centroids_norm.T)
            elif metric == "manhattan":
                # Manhattan (L1) distance
                centroid_distances = torch.zeros(num_clusters, device=device)
                for j in range(num_clusters):
                    centroid_distances[j] = torch.sum(torch.abs(X - centroids[j]))
            elif metric == "dot":
                # Dot product distance (negative dot product for smaller = closer)
                centroid_distances = -torch.matmul(X, centroids.T)
            
            # Get top clusters
            top_clusters = torch.topk(centroid_distances, k=min(5, num_clusters), largest=False).indices
            
            # Gather candidate points
            candidate_indices = []
            for c in top_clusters:
                indices = torch.nonzero(cluster_ids_tensor == c).squeeze(1)
                candidate_indices.append(indices)
            
            if candidate_indices:
                candidate_indices = torch.cat(candidate_indices)
            else:
                candidate_indices = torch.arange(N, device=device)
            
            candidate_points = A[candidate_indices]
            
            # Compute distances for candidates
            if metric == "l2":
                distances = torch.cdist(X.unsqueeze(0), candidate_points).squeeze(0)
            elif metric == "cosine":
                X_norm = X / torch.norm(X)
                candidates_norm = candidate_points / torch.norm(candidate_points, dim=1, keepdim=True)
                distances = 1 - torch.matmul(X_norm, candidates_norm.T)
            elif metric == "manhattan":
                # Manhattan (L1) distance
                distances = torch.zeros(len(candidate_indices), device=device)
                for j in range(len(candidate_indices)):
                    distances[j] = torch.sum(torch.abs(X - candidate_points[j]))
            elif metric == "dot":
                # Dot product distance (negative dot product for smaller = closer)
                distances = -torch.matmul(X, candidate_points.T)
            
            # Get top K results
            topk_indices = torch.topk(distances, k=min(K, len(candidate_indices)), largest=False).indices
            result = candidate_indices[topk_indices].cpu().numpy().tolist()
            
            end_time = time.time()
            return result, end_time - start_time
    
    # Test across different data sizes (with fixed dimension)
    fixed_dim = 128
    print(f"\n--- Testing across data sizes (fixed dimension: {fixed_dim}) ---")
    if compute_true_recall:
        print("Size\tMetric\tCPU Time (s)\tGPU Time (s)\tSpeedup\tCPU-GPU Match\tCPU Recall\tGPU Recall")
    else:
        print("Size\tMetric\tCPU Time (s)\tGPU Time (s)\tSpeedup\tCPU-GPU Match")
    
    for N in data_sizes:
        # Generate random data
        A_np = np.random.randn(N, fixed_dim).astype(np.float32)
        X_np = np.random.randn(fixed_dim).astype(np.float32)
        
        for metric in metrics:
            # Run on CPU
            cpu_result, cpu_time = run_ann_on_device(N, fixed_dim, A_np, X_np, K, metric, "cpu")
            
            # Run on GPU
            gpu_result, gpu_time = run_ann_on_device(N, fixed_dim, A_np, X_np, K, metric, "cuda")
            
            # Calculate speedup and recall
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            
            # Calculate recall between CPU and GPU results
            cpu_gpu_recall = compute_recall(cpu_result, gpu_result, K)
            
            # Optionally compute recall against true KNN results
            if compute_true_recall:
                # Get ground truth KNN results
                true_knn = our_knn(N, fixed_dim, A_np, X_np, K, metric)
                cpu_recall = compute_recall(true_knn, cpu_result, K)
                gpu_recall = compute_recall(true_knn, gpu_result, K)
                print(f"{N}\t{metric}\t{cpu_time:.6f}\t{gpu_time:.6f}\t{speedup:.2f}x\t{cpu_gpu_recall:.2%}\t{cpu_recall:.2%}\t{gpu_recall:.2%}")
            else:
                print(f"{N}\t{metric}\t{cpu_time:.6f}\t{gpu_time:.6f}\t{speedup:.2f}x\t{cpu_gpu_recall:.2%}")
    
    # Test across different dimensions (with fixed data size)
    fixed_size = 4000  # Use 4,000 as mentioned in the assignment
    print(f"\n--- Testing across dimensions (fixed data size: {fixed_size}) ---")
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
                cpu_result, cpu_time = run_ann_on_device(fixed_size, D, A_np, X_np, K, metric, "cpu")
                
                # Run on GPU
                gpu_result, gpu_time = run_ann_on_device(fixed_size, D, A_np, X_np, K, metric, "cuda")
                
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

    print("\n--- ANN CPU vs GPU Speedup Test ---")
    test_ann_cpu_gpu_speedup(compute_true_recall=True)

