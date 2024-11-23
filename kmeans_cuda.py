import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA 驱动
from pycuda.compiler import SourceModule
import math

# 定义 CUDA 内核，使用 C++ 代码，以字符串形式
kernels = """
__global__ void update_centroids_kernel(float *X, int *labels, float *new_centroids, int *cluster_sizes, int n_samples, int n_features) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx < n_samples) {
        int label = labels[sample_idx];
        atomicAdd(&cluster_sizes[label], 1);
        
        for (int feat_idx = 0; feat_idx < n_features; feat_idx++) {
            atomicAdd(&new_centroids[label * n_features + feat_idx], X[sample_idx * n_features + feat_idx]);
        }
    }
}

__global__ void compute_centroids_mean_kernel(float *new_centroids, int *cluster_sizes, int n_clusters, int n_features) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (cluster_idx < n_clusters && feat_idx < n_features) {
        int size = cluster_sizes[cluster_idx];
        if (size > 0) {
            new_centroids[cluster_idx * n_features + feat_idx] /= size;
        }
    }
}
"""

mod = SourceModule(kernels)
update_centroids_kernel = mod.get_function("update_centroids_kernel")
compute_centroids_mean_kernel = mod.get_function("compute_centroids_mean_kernel")

class CUDAKMeans:
    """使用 PyCUDA 加速的 K-means 算法"""
    
    @staticmethod
    def update_centroids(X, labels, n_clusters):
        """使用 CUDA 加速更新质心
        
        参数：
            X: 输入数据，形状为 (n_samples, n_features)
            labels: 聚类标签，形状为 (n_samples,)
            n_clusters: 聚类数量
                
        返回：
            new_centroids: 更新后的质心，形状为 (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        
        # 转换数据类型
        X = X.astype(np.float32)
        labels = labels.astype(np.int32)
        
        # 分配 GPU 内存
        X_gpu = cuda.mem_alloc(X.nbytes)
        labels_gpu = cuda.mem_alloc(labels.nbytes)
        new_centroids_gpu = cuda.mem_alloc(n_clusters * n_features * X.dtype.itemsize)
        cluster_sizes_gpu = cuda.mem_alloc(n_clusters * np.int32().itemsize)
        
        # 初始化 new_centroids 和 cluster_sizes 为零
        cuda.memset_d32(new_centroids_gpu, 0, n_clusters * n_features)
        cuda.memset_d32(cluster_sizes_gpu, 0, n_clusters)
        
        # 将数据复制到 GPU
        cuda.memcpy_htod(X_gpu, X)
        cuda.memcpy_htod(labels_gpu, labels)
        
        # 配置网格和块尺寸
        threads_per_block = 256
        blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block
        
        # 运行第一个内核以累加样本
        update_centroids_kernel(
            X_gpu, labels_gpu, new_centroids_gpu, cluster_sizes_gpu,
            np.int32(n_samples), np.int32(n_features),
            block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1)
        )
        
        # 为均值计算配置网格
        threads_per_block_2d = (16, 16)
        blocks_per_grid_2d = (
            (n_clusters + threads_per_block_2d[0] - 1) // threads_per_block_2d[0],
            (n_features + threads_per_block_2d[1] - 1) // threads_per_block_2d[1]
        )
        
        # 运行第二个内核以计算均值
        compute_centroids_mean_kernel(
            new_centroids_gpu, cluster_sizes_gpu,
            np.int32(n_clusters), np.int32(n_features),
            block=(threads_per_block_2d[0], threads_per_block_2d[1], 1),
            grid=(blocks_per_grid_2d[0], blocks_per_grid_2d[1], 1)
        )
        
        # 将结果复制回 CPU
        new_centroids = np.empty((n_clusters, n_features), dtype=np.float32)
        cuda.memcpy_dtoh(new_centroids, new_centroids_gpu)
        
        return new_centroids
