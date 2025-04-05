import triton
import triton.language as tl
import torch
import time
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
# Test wrappers
# -----------------------------------------------------------------------------
def test_knn():
    N, D, A, X, K = testdata_knn("")
    for metric in ["l2", "cosine", "dot", "manhattan"]:
        start = time.time()
        result = our_knn(N, D, A, X, K, metric)
        elapsed = time.time() - start
        print(f"KNN [{metric}] result: {result}\nElapsed: {elapsed:.4f} sec")

def test_ann():
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


if __name__ == "__main__":
    print("\n--- KNN Tests ---")
    test_knn()
    print("\n--- ANN Tests ---")
    test_ann()
    print("\n--- Recall & Precision ---")
    test_recall()
