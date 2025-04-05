import time
import numpy as np
from task import our_knn, our_kmeans, our_ann, compute_recall
from test import testdata_knn, testdata_kmeans, testdata_ann

def run_knn_test(test_file, metrics=["l2", "cosine", "dot", "manhattan"]):
    print(f"\nRunning KNN test with {test_file}")
    N, D, A, X, K = testdata_knn(test_file)
    print(f"Data shape: {N} vectors of dimension {D}, searching for top {K}")
    
    for metric in metrics:
        # GPU implementation
        start = time.time()
        result = our_knn(N, D, A, X, K, metric)
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
                distances.append(1.0 - dot / (norm_a * norm_x))
        elif metric == "dot":
            for i in range(N):
                distances.append(-np.sum(A[i] * X))
        
        top_indices = np.argsort(distances)[:K].tolist()
        cpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        
        print(f"  {metric} metric:")
        print(f"    GPU time: {gpu_time:.6f} sec")
        print(f"    CPU time: {cpu_time:.6f} sec")
        print(f"    Speedup: {speedup:.2f}x")

def run_kmeans_test(test_file, metrics=["l2", "cosine"]):
    print(f"\nRunning KMeans test with {test_file}")
    N, D, A, K = testdata_kmeans(test_file)
    print(f"Data shape: {N} vectors of dimension {D}, {K} clusters")
    
    for metric in metrics:
        start = time.time()
        result = our_kmeans(N, D, A, K, metric)
        elapsed = time.time() - start
        clusters = {}
        for i, cluster_id in enumerate(result):
            if cluster_id not in clusters:
                clusters[cluster_id] = 0
            clusters[cluster_id] += 1
        
        print(f"  {metric} metric:")
        print(f"    Time: {elapsed:.6f} sec")
        print(f"    Cluster distribution: {clusters}")

def run_ann_test(test_file, metrics=["l2", "cosine"]):
    print(f"\nRunning ANN test with {test_file}")
    N, D, A, X, K = testdata_ann(test_file)
    print(f"Data shape: {N} vectors of dimension {D}, searching for top {K}")
    
    for metric in metrics:
        # Run exact KNN first
        knn_start = time.time()
        knn_result = our_knn(N, D, A, X, K, metric)
        knn_time = time.time() - knn_start
        
        # Run ANN
        ann_start = time.time()
        ann_result = our_ann(N, D, A, X, K, metric)
        ann_time = time.time() - ann_start
        
        # Calculate recall
        recall = compute_recall(knn_result, ann_result, K)
        speedup = knn_time / ann_time if ann_time > 0 else float('inf')
        
        print(f"  {metric} metric:")
        print(f"    KNN time: {knn_time:.6f} sec")
        print(f"    ANN time: {ann_time:.6f} sec")
        print(f"    Speedup: {speedup:.2f}x")
        print(f"    Recall: {recall:.2%}")

if __name__ == "__main__":
    # KNN tests
    knn_tests = [
        "test_data\large_128d_knn.json",
        "test_data\medium_1024d_knn.json",
        "test_data\medium_128d_knn.json",
        "test_data\medium_2d_knn.json",
        "test_data\small_1024d_knn.json",
        "test_data\small_128d_knn.json",
        "test_data\small_2d_knn.json",
    ]
    
    for test in knn_tests:
        run_knn_test(test)
    
    # KMeans tests
    kmeans_tests = [
        "test_data\large_128d_kmeans.json",
        "test_data\medium_1024d_kmeans.json",
        "test_data\medium_128d_kmeans.json",
        "test_data\medium_2d_kmeans.json",
        "test_data\small_1024d_kmeans.json",
        "test_data\small_128d_kmeans.json",
        "test_data\small_2d_kmeans.json",
    ]
    
    for test in kmeans_tests:
        run_kmeans_test(test)
    
    # ANN tests (reuse KNN test files)
    for test in knn_tests:
        run_ann_test(test)
