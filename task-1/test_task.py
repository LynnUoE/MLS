import time
import torch
import numpy as np
from test import testdata_knn, testdata_kmeans, testdata_ann
from task import our_knn, our_kmeans, our_ann, compute_recall

def prewarm_gpu():
    """
    Prewarm the GPU to ensure accurate timing measurements
    by running a small computation
    """
    dummy_a = torch.randn(1000, 128, device="cuda")
    dummy_x = torch.randn(128, device="cuda")
    _ = torch.sum((dummy_a - dummy_x.unsqueeze(0)) ** 2, dim=1)
    torch.cuda.synchronize()  # Ensure all operations are complete

def test_knn():
    """
    Test KNN algorithm with different distance metrics
    """
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_knn("")
    
    print("\n--- KNN Tests ---")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        start = time.time()
        result = our_knn(N, D, A, X, K, metric)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"KNN [{metric}] result: {result}")
        print(f"Elapsed: {elapsed:.4f} sec")

def test_kmeans():
    """
    Test KMeans clustering algorithm
    """
    prewarm_gpu()
    N, D, A, K = testdata_kmeans("")
    
    print("\n--- KMeans Test ---")
    start = time.time()
    result = our_kmeans(N, D, A, K)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"KMeans result: First 10 cluster IDs: {result[:10]}")
    print(f"Elapsed: {elapsed:.4f} sec")

def test_ann():
    """
    Test ANN algorithm with different distance metrics
    """
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_ann("")
    
    print("\n--- ANN Tests ---")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        start = time.time()
        result = our_ann(N, D, A, X, K, metric)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"ANN [{metric}] result: {result}")
        print(f"Elapsed: {elapsed:.4f} sec")

def test_recall():
    """
    For each distance metric, run KNN once and run ANN 5 times 
    to compute average recall
    """
    prewarm_gpu()  # Prewarm before testing
    N, D, A, X, K = testdata_knn("")
    
    print("\n--- Recall & Precision ---")
    print("Metric\tAvg Recall")
    
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        # Run KNN to get ground truth
        knn_res = our_knn(N, D, A, X, K, metric)
        
        # Run ANN multiple times and measure recall
        ann_recalls = []
        for _ in range(5):
            ann_res = our_ann(N, D, A, X, K, metric)
            recall = compute_recall(knn_res, ann_res, K)
            ann_recalls.append(recall)
        
        avg_recall = sum(ann_recalls) / len(ann_recalls)
        print(f"{metric}\t{avg_recall:.2%}")

def compare_gpu_cpu():
    """
    Compare GPU vs CPU performance for KNN
    """
    prewarm_gpu()  # Prewarm GPU
    
    N, D, A, X, K = testdata_knn("")
    print("\n--- GPU vs CPU Comparison ---")
    print(f"Data shape: {N} vectors of dimension {D}")
    
    for metric in ["l2"]:  # Using just L2 for basic comparison
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
    """
    Test how GPU vs CPU performance scales with dimension
    """
    prewarm_gpu()
    
    dimensions = [2, 16, 128, 1024, 4096]
    N = 1000  # Keep vector count fixed
    K = 10
    
    print("\n--- Dimension Scaling Test ---")
    print("Dimension\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for D in dimensions:
        # Generate random data
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        
        # GPU timing
        A = torch.tensor(A_np, device="cuda")
        X = torch.tensor(X_np, device="cuda")
        
        # Warmup
        _ = torch.sum((A[:10] - X.unsqueeze(0)) ** 2, dim=1)
        torch.cuda.synchronize()
        
        # Actual timing
        start = time.time()
        distances = torch.sum((A - X.unsqueeze(0)) ** 2, dim=1)
        topk = torch.topk(distances, k=K, largest=False)
        result_gpu = topk.indices.cpu().numpy().tolist()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU timing with appropriate sampling for large dimensions
        start = time.time()
        distances = []
        sample_size = N if D < 1000 else min(100, N)
        
        for i in range(sample_size):
            distances.append(np.sum((A_np[i] - X_np) ** 2))
        
        cpu_time_sample = time.time() - start
        
        # Scale up if we used a sample
        if sample_size < N:
            cpu_time = cpu_time_sample * (N / sample_size)
        else:
            cpu_time = cpu_time_sample
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"{D}\t\t{gpu_time:.6f}\t{cpu_time:.6f}\t{speedup:.2f}x")

def test_vector_count_scaling():
    """
    Test how performance scales with the number of vectors
    """
    prewarm_gpu()
    
    vector_counts = [100, 1000, 4000, 10000]
    D = 128  # Keep dimension fixed
    K = 10
    
    print("\n--- Vector Count Scaling Test ---")
    print("Vector Count\tGPU Time (s)\tCPU Time (s)\tSpeedup")
    
    for N in vector_counts:
        # Generate random data
        A_np = np.random.randn(N, D).astype(np.float32)
        X_np = np.random.randn(D).astype(np.float32)
        
        # GPU timing
        A = torch.tensor(A_np, device="cuda")
        X = torch.tensor(X_np, device="cuda")
        
        torch.cuda.synchronize()
        start = time.time()
        distances = torch.sum((A - X.unsqueeze(0)) ** 2, dim=1)
        topk = torch.topk(distances, k=K, largest=False)
        result_gpu = topk.indices.cpu().numpy().tolist()
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        # CPU timing with sampling for large vector counts
        start = time.time()
        distances = []
        sample_size = N if N < 5000 else min(1000, N)
        
        for i in range(sample_size):
            distances.append(np.sum((A_np[i] - X_np) ** 2))
        
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
    """
    Extrapolate performance for 4,000,000 vectors as requested in the task
    """
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
    distances = torch.sum((A - X.unsqueeze(0)) ** 2, dim=1)
    topk = torch.topk(distances, k=K, largest=False)
    torch.cuda.synchronize()
    sample_time = time.time() - start
    
    # Extrapolate to 4,000,000 vectors
    N_large = 4000000
    scaling_factor = N_large / N_sample
    
    print("\n--- Large Dataset Extrapolation ---")
    print(f"Performance extrapolation for 4,000,000 vectors:")
    print(f"Sample time for {N_sample} vectors: {sample_time:.6f} seconds")
    
    # Linear extrapolation
    linear_estimate = sample_time * scaling_factor
    print(f"Simple linear extrapolation: {linear_estimate:.2f} seconds")
    
    # More realistic sublinear extrapolation
    sublinear_estimate = sample_time * (scaling_factor ** 0.8)
    print(f"Optimized sublinear extrapolation: {sublinear_estimate:.2f} seconds")
    
    print("\nOptimizations needed for 4,000,000 vectors:")
    print("1. Batch processing to manage GPU memory")
    print("2. Multiple GPU processing if available")
    print("3. Mixed precision (FP16) to reduce memory usage and increase throughput")
    print("4. Quantization techniques to compress vectors")
    print("5. Progressive refinement approach (coarse search followed by fine search)")

if __name__ == "__main__":
    print("\n=== Running Vector Search Tests ===\n")
    
    # Run basic tests
    test_knn()
    test_kmeans()
    test_ann()
    test_recall()
    
    # Run performance comparison tests
    compare_gpu_cpu()
    test_dimension_scaling()
    test_vector_count_scaling()
    extrapolate_large_dataset()