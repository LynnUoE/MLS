import numpy as np
import json
import os
import time

def generate_data_files(output_dir="test_data"):
    """Generate data files for testing KNN, KMeans, and ANN specifically tailored to answer report questions"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define test cases that address specific report questions
    test_cases = [
        # Question 2: Dimension comparison (2 vs 2^15)
        {"n": 1000, "d": 2, "k": 10, "name": "dim_2"},
        {"n": 1000, "d": 32768, "k": 10, "name": "dim_2_15"},  # 2^15 = 32768
        
        # Question 5: 4,000 vectors
        {"n": 4000, "d": 128, "k": 10, "name": "vectors_4k"},
        
        # Question 5: 4,000,000 vectors (generate a smaller file for testing,
        # but with the right configuration for 4M vectors)
        {"n": 4000000, "d": 128, "k": 10, "name": "vectors_4m_config"},
        {"n": 10000, "d": 128, "k": 10, "name": "vectors_10k_sample"}  # Sample for actual testing
    ]
    
    for case in test_cases:
        n = case["n"]  # Number of vectors
        d = case["d"]  # Dimension
        k = case["k"]  # K value
        name = case["name"]
        
        print(f"Generating data for {name}: {n} vectors of dimension {d}")
        
        # For very large configurations, we'll create just the JSON config 
        # but use a smaller sample for actual testing
        if n > 100000:  # Don't generate full data for very large cases
            if "config" in name:
                # Just create the config file, not the actual data
                config = {
                    "n": n,
                    "d": d,
                    "a_file": f"{output_dir}/{name.replace('_config', '')}_vectors.txt",
                    "x_file": f"{output_dir}/{name.replace('_config', '')}_query.txt",
                    "k": k
                }
                
                json_file = os.path.join(output_dir, f"{name}_test.json")
                with open(json_file, "w") as f:
                    json.dump(config, f, indent=2)
                
                print(f"Created configuration file for large dataset: {json_file}")
                continue
        
        # Generate vector collection A
        start_time = time.time()
        print(f"  Generating {n} vectors...")
        
        # For very large dimensions, use a more memory-efficient approach
        if d > 10000:
            # Generate and save in chunks to avoid memory issues
            chunk_size = 100  # Adjust based on your system's memory
            a_file = os.path.join(output_dir, f"{name}_vectors.txt")
            
            with open(a_file, 'w') as f:
                for i in range(0, n, chunk_size):
                    end_idx = min(i + chunk_size, n)
                    chunk_n = end_idx - i
                    chunk = np.random.randn(chunk_n, d).astype(np.float32)
                    np.savetxt(f, chunk, fmt='%.6f')
                    print(f"  Generated vectors {i} to {end_idx-1}")
        else:
            # Standard approach for reasonable dimensions
            A = np.random.randn(n, d).astype(np.float32)
            a_file = os.path.join(output_dir, f"{name}_vectors.txt")
            np.savetxt(a_file, A)
        
        # Generate query vector X
        X = np.random.randn(d).astype(np.float32)
        x_file = os.path.join(output_dir, f"{name}_query.txt")
        np.savetxt(x_file, X)
        
        print(f"  Data generation took {time.time() - start_time:.2f} seconds")
        
        # Create JSON config file
        config = {
            "n": n,
            "d": d,
            "a_file": a_file,
            "x_file": x_file,
            "k": k
        }
        
        json_file = os.path.join(output_dir, f"{name}_test.json")
        with open(json_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Created test file: {json_file}")

def create_test_script():
    """Create a test script that specifically addresses the report questions"""
    
    with open("task1_test_report.py", "w") as f:
        f.write('''import os
import time
import numpy as np
import torch
from task1.task import our_knn, our_kmeans, our_ann, compute_recall, compute_distance
from task1.test import testdata_knn, testdata_kmeans, testdata_ann

def test_distance_functions():
    """
    Question 1: How did you implement four distinct distance functions on the GPU?
    Test all distance functions implementation
    """
    print("\\n=== Testing Distance Functions ===")
    # Create some sample data
    d = 128
    n = 1000
    A = np.random.randn(n, d).astype(np.float32)
    X = np.random.randn(d).astype(np.float32)
    
    # Move to GPU
    A_gpu = torch.tensor(A, device="cuda")
    X_gpu = torch.tensor(X, device="cuda")
    
    # Test each distance function
    metrics = ["l2", "cosine", "dot", "manhattan"]
    for metric in metrics:
        start = time.time()
        distances = compute_distance(A_gpu, X_gpu, metric=metric)
        elapsed = time.time() - start
        
        print(f"  {metric.upper()} Distance:")
        print(f"    Time: {elapsed:.6f} seconds")
        print(f"    First 5 distances: {distances[:5].cpu().numpy()}")
        print()

def test_dimension_speedup():
    """
    Question 2: What is the speed advantage of the GPU over the CPU version 
    when the dimension is 2? Additionally, what is the speed advantage when 
    the dimension is 2^15?
    """
    print("\\n=== Testing Dimension Speedup ===")
    
    # Test for dimension 2
    test_file = "test_data/dim_2_test.json"
    if os.path.exists(test_file):
        N, D, A, X, K = testdata_knn(test_file)
        
        print(f"Testing with dimension {D}:")
        
        # GPU version
        start = time.time()
        result_gpu = our_knn(N, D, A, X, K, "l2")
        gpu_time = time.time() - start
        
        # CPU version
        start = time.time()
        distances = []
        for i in range(N):
            dist = np.sqrt(np.sum((A[i] - X) ** 2))
            distances.append(dist)
        sorted_indices = np.argsort(distances)
        result_cpu = sorted_indices[:K].tolist()
        cpu_time = time.time() - start
        
        print(f"  GPU time: {gpu_time:.6f} seconds")
        print(f"  CPU time: {cpu_time:.6f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    else:
        print(f"Test file {test_file} not found. Skipping dimension 2 test.")
    
    # Test for dimension 2^15
    test_file = "test_data/dim_2_15_test.json"
    if os.path.exists(test_file):
        N, D, A, X, K = testdata_knn(test_file)
        
        print(f"\\nTesting with dimension {D}:")
        
        # GPU version
        start = time.time()
        result_gpu = our_knn(N, D, A, X, K, "l2")
        gpu_time = time.time() - start
        
        # CPU version - for very high dimensions, just estimate with a small sample
        sample_size = min(100, N)
        start = time.time()
        distances = []
        for i in range(sample_size):
            dist = np.sqrt(np.sum((A[i] - X) ** 2))
            distances.append(dist)
        cpu_time_sample = time.time() - start
        
        # Extrapolate to full dataset
        cpu_time_estimated = cpu_time_sample * (N / sample_size)
        
        print(f"  GPU time: {gpu_time:.6f} seconds")
        print(f"  Estimated CPU time (based on {sample_size} samples): {cpu_time_estimated:.6f} seconds")
        print(f"  Estimated speedup: {cpu_time_estimated/gpu_time:.2f}x")
    else:
        print(f"Test file {test_file} not found. Skipping dimension 2^15 test.")

def test_top_k_algorithm():
    """
    Question 3: Please provide a detailed description of your Top K algorithm.
    Question 4: What steps did you undertake to implement the Top K on the GPU? 
    How do you manage data within GPU memory?
    
    Test the performance of Top K retrieval
    """
    print("\\n=== Testing Top K Algorithm ===")
    
    test_file = "test_data/vectors_4k_test.json"
    if os.path.exists(test_file):
        N, D, A, X, K = testdata_knn(test_file)
        
        print(f"Testing Top K with {N} vectors of dimension {D}, K={K}")
        
        # Run KNN with profiling
        start = time.time()
        result = our_knn(N, D, A, X, K, "l2")
        elapsed = time.time() - start
        
        print(f"  Top K computation completed in {elapsed:.6f} seconds")
        print(f"  First 5 indices: {result[:5]}")
    else:
        print(f"Test file {test_file} not found. Skipping Top K test.")

def test_large_vectors():
    """
    Question 5: When processing 4,000 vectors, how many seconds does the operation take? 
    Furthermore, when handling 4,000,000 vectors, what modifications did you implement 
    to ensure the effective functioning of your code?
    """
    print("\\n=== Testing Large Vectors ===")
    
    # Test with 4,000 vectors
    test_file = "test_data/vectors_4k_test.json"
    if os.path.exists(test_file):
        N, D, A, X, K = testdata_knn(test_file)
        
        print(f"Testing with {N} vectors of dimension {D}:")
        
        start = time.time()
        result = our_knn(N, D, A, X, K, "l2")
        elapsed = time.time() - start
        
        print(f"  Operation took {elapsed:.6f} seconds")
    else:
        print(f"Test file {test_file} not found. Skipping 4,000 vectors test.")
    
    # Test with sample of large dataset to demonstrate scalability
    test_file = "test_data/vectors_10k_sample_test.json"
    if os.path.exists(test_file):
        N, D, A, X, K = testdata_knn(test_file)
        
        print(f"\\nTesting with {N} vectors sample (for 4,000,000 vectors):")
        
        start = time.time()
        result = our_knn(N, D, A, X, K, "l2")
        elapsed = time.time() - start
        
        # Extrapolate performance to 4M vectors
        extrapolation_factor = 4000000 / N
        estimated_time = elapsed * extrapolation_factor
        
        print(f"  Sample operation took {elapsed:.6f} seconds")
        print(f"  Estimated time for 4,000,000 vectors: {estimated_time:.2f} seconds")
        print("  For 4,000,000 vectors, you would need to implement:")
        print("    - Batch processing to handle GPU memory constraints")
        print("    - Parallel processing across multiple GPUs if available")
        print("    - Optimizing memory usage with mixed precision or quantization")
    else:
        print(f"Test file {test_file} not found. Skipping large vectors test.")

if __name__ == "__main__":
    print("=== Task 1 Report Tests ===")
    
    # Run all tests that address the specific report questions
    test_distance_functions()
    test_dimension_speedup()
    test_top_k_algorithm()
    test_large_vectors()
    
    print("\\nAll tests completed.")
''')
    
    print("Created test script: task1_test_report.py")

if __name__ == "__main__":
    generate_data_files()
    create_test_script()
    
    print("\nTest data generation complete.")
    print("You can now run the tests with: python task1_test_report.py")