import numpy as np
import json
import os

def generate_data_files(output_dir="test_data"):
    """Generate data files for testing KNN, KMeans, and ANN"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test cases with different dimensions and vector counts
    test_cases = [
        # Small cases
        {"n": 100, "d": 2, "k": 5, "name": "small_2d"},
        {"n": 100, "d": 128, "k": 5, "name": "small_128d"},
        {"n": 100, "d": 1024, "k": 5, "name": "small_1024d"},
        
        # Medium cases
        {"n": 4000, "d": 2, "k": 10, "name": "medium_2d"},
        {"n": 4000, "d": 128, "k": 10, "name": "medium_128d"},
        {"n": 4000, "d": 1024, "k": 10, "name": "medium_1024d"},
        
        # Large case (for reference, not for actual testing)
        {"n": 10000, "d": 128, "k": 20, "name": "large_128d"}
    ]
    
    for case in test_cases:
        n = case["n"]  # Number of vectors
        d = case["d"]  # Dimension
        k = case["k"]  # K value
        name = case["name"]
        
        print(f"Generating data for {name}: {n} vectors of dimension {d}")
        
        # Generate vector collection A
        A = np.random.randn(n, d).astype(np.float32)
        a_file = os.path.join(output_dir, f"{name}_vectors.txt")
        np.savetxt(a_file, A)
        
        # Generate query vector X
        X = np.random.randn(d).astype(np.float32)
        x_file = os.path.join(output_dir, f"{name}_query.txt")
        np.savetxt(x_file, X)
        
        # Create JSON config file for KNN and ANN
        knn_config = {
            "n": n,
            "d": d,
            "a_file": a_file,
            "x_file": x_file,
            "k": k
        }
        
        with open(os.path.join(output_dir, f"{name}_knn.json"), "w") as f:
            json.dump(knn_config, f, indent=2)
        
        # Create JSON config file for KMeans
        kmeans_config = {
            "n": n,
            "d": d,
            "a_file": a_file,
            "k": min(k, n // 10)  # K for KMeans is cluster count, keep it reasonable
        }
        
        with open(os.path.join(output_dir, f"{name}_kmeans.json"), "w") as f:
            json.dump(kmeans_config, f, indent=2)

def run_tests(test_dir="test_data"):
    """Generate a script to run tests with the generated data"""
    
    # Find all test config files
    knn_tests = [f for f in os.listdir(test_dir) if f.endswith("_knn.json")]
    kmeans_tests = [f for f in os.listdir(test_dir) if f.endswith("_kmeans.json")]
    
    # Generate test runner script
    with open("run_tests.py", "w") as f:
        f.write("""import time
import numpy as np
from task import our_knn, our_kmeans, our_ann, compute_recall
from test import testdata_knn, testdata_kmeans, testdata_ann

def run_knn_test(test_file, metrics=["l2", "cosine", "dot", "manhattan"]):
    print(f"\\nRunning KNN test with {test_file}")
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
    print(f"\\nRunning KMeans test with {test_file}")
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
    print(f"\\nRunning ANN test with {test_file}")
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
""")
        
        for test in knn_tests:
            f.write(f'        "{os.path.join(test_dir, test)}",\n')
        
        f.write("""    ]
    
    for test in knn_tests:
        run_knn_test(test)
    
    # KMeans tests
    kmeans_tests = [
""")
        
        for test in kmeans_tests:
            f.write(f'        "{os.path.join(test_dir, test)}",\n')
        
        f.write("""    ]
    
    for test in kmeans_tests:
        run_kmeans_test(test)
    
    # ANN tests (reuse KNN test files)
    for test in knn_tests:
        run_ann_test(test)
""")

if __name__ == "__main__":
    generate_data_files()
    run_tests()
    print("Test data and runner script generated successfully.")
    print("Run 'python run_tests.py' to execute the tests.")