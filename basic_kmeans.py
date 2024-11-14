import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import silhouette_score
from datetime import datetime
import time
from contextlib import contextmanager

class Timer:
    """计时器类，用于记录和管理各个步骤的时间"""
    def __init__(self):
        self.times = {}
        self.start_times = {}

    @contextmanager
    def timer(self, name):
        """上下文管理器，用于计时代码块的执行时间"""
        try:
            self.start_times[name] = time.time()
            yield
        finally:
            end_time = time.time()
            duration = end_time - self.start_times[name]
            if name in self.times:
                if isinstance(self.times[name], list):
                    self.times[name].append(duration)
                else:
                    self.times[name] = [self.times[name], duration]
            else:
                self.times[name] = duration

    def get_report(self):
        """生成计时报告"""
        report = "\nTiming Report:\n" + "="*50 + "\n"
        for name, duration in self.times.items():
            if isinstance(duration, list):
                avg_time = np.mean(duration)
                total_time = np.sum(duration)
                report += f"{name}:\n"
                report += f"  Average time: {avg_time:.3f}s\n"
                report += f"  Total time: {total_time:.3f}s\n"
                report += f"  Iterations: {len(duration)}\n"
            else:
                report += f"{name}: {duration:.3f}s\n"
        return report

# 创建全局计时器实例
timer = Timer()

def load_mnist(n_samples=5000):
    """Load and preprocess MNIST dataset"""
    with timer.timer("Data Loading"):
        print("Loading MNIST dataset...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # 随机选择n_samples个样本
        idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X = X[idx]
        y = y[idx]
        
        # 标准化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X, y

class KMeans:
    def __init__(self, n_clusters=10, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None
        self.inertia_history_ = []
        self.iteration_times_ = []
        self.n_iters_ = 0
        
    def initialize_centroids(self, X):
        """随机初始化质心"""
        with timer.timer("Centroid Initialization"):
            n_samples, n_features = X.shape
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[idx]
        
    def get_distance(self, X):
        """计算每个样本到所有质心的距离"""
        with timer.timer("Distance Calculation"):
            distances = np.zeros((X.shape[0], self.n_clusters))
            for k in range(self.n_clusters):
                distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        return distances
    
    def fit(self, X):
        """训练K-means模型"""
        with timer.timer("K-means Training"):
            self.initialize_centroids(X)
            
            for i in tqdm(range(self.max_iters), desc="Training K-means"):
                with timer.timer("Single Iteration"):
                    # 计算距离并分配簇
                    distances = self.get_distance(X)
                    self.labels_ = np.argmin(distances, axis=1)
                    
                    # 计算inertia
                    current_inertia = np.sum(np.min(distances, axis=1))
                    self.inertia_history_.append(current_inertia)
                    
                    # 更新质心
                    with timer.timer("Centroid Update"):
                        new_centroids = np.array([X[self.labels_ == k].mean(axis=0) 
                                                for k in range(self.n_clusters)])
                    
                    # 检查收敛
                    if np.all(self.centroids == new_centroids):
                        break
                        
                    self.centroids = new_centroids
                    self.n_iters_ = i + 1
            
        return self.labels_


def visualize_centroids(centroids, cluster_sizes, cluster_purities):
    """可视化质心（增强版）"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for idx, (centroid, size, purity) in enumerate(zip(centroids, cluster_sizes, cluster_purities)):
        img = centroid.reshape(28, 28)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Cluster {idx}\nSize: {size} samples\nPurity: {purity:.1f}%', 
                          fontsize=10, pad=10)
    
    plt.suptitle("Learned Cluster Centers with Statistics", fontsize=16, y=1.02)
    fig.text(0.5, -0.02, 
             "Purity indicates the percentage of the most common digit in each cluster",
             ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.savefig('cluster_centers.png', dpi=300, bbox_inches='tight')

def plot_loss_curve(inertia_history, iteration_times):
    """绘制增强版损失曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(inertia_history, 'b-', linewidth=2, label='Inertia')
    ax1.set_title('K-means Convergence Curve', fontsize=12)
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 计算改进百分比
    total_improvement = ((inertia_history[0] - inertia_history[-1]) / inertia_history[0]) * 100
    ax1.text(0.02, 0.98, f'Total improvement: {total_improvement:.1f}%',
             transform=ax1.transAxes, verticalalignment='top')
    
    # 迭代时间曲线
    ax2.plot(iteration_times, 'r-', linewidth=2, label='Iteration Time')
    ax2.set_title('Iteration Time per Step', fontsize=12)
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Time (seconds)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 添加平均时间标注
    avg_time = np.mean(iteration_times)
    ax2.axhline(y=avg_time, color='g', linestyle='--', alpha=0.5)
    ax2.text(0.02, 0.98, f'Average time: {avg_time:.3f}s',
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)

def plot_cluster_distribution(labels, y, cluster_sizes):
    """绘制增强版聚类分布热力图"""
    cluster_dist = np.zeros((10, 10))
    
    for cluster in range(10):
        cluster_labels = y[labels == cluster].astype(int)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        for label, count in zip(unique, counts):
            cluster_dist[cluster, label] = count
            
    # 归一化
    cluster_dist_normalized = cluster_dist / cluster_dist.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(cluster_dist_normalized, annot=True, fmt='.2%', cmap='YlOrRd')
    
    # 添加簇大小标注
    plt.title('Distribution of True Labels in Each Cluster\n' + 
             'Numbers show percentage of each digit in cluster', 
             fontsize=14, pad=20)
    
    # 在y轴标签旁添加簇大小信息
    ylabels = [f'Cluster {i}\n(n={size})' for i, size in enumerate(cluster_sizes)]
    plt.yticks(np.arange(10) + 0.5, ylabels, rotation=0)
    
    plt.xlabel('True Digit Label', fontsize=12)
    
    # 添加说明文本
    plt.figtext(0.99, 0.01, 
                'Darker colors indicate higher concentration of a digit in a cluster',
                ha='right', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('cluster_distribution.png')

def visualize_pca_clusters(X, labels, y, silhouette_avg):
    """使用PCA进行增强版聚类可视化"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 聚类结果
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax1.set_title('Clustering Results\n' + 
                  f'Silhouette Score: {silhouette_avg:.3f}', fontsize=14)
    ax1.set_xlabel(f'First Principal Component\nExplained Variance: {pca.explained_variance_ratio_[0]:.1%}')
    ax1.set_ylabel(f'Second Principal Component\nExplained Variance: {pca.explained_variance_ratio_[1]:.1%}')
    plt.colorbar(scatter1, ax=ax1, label='Cluster Label')
    
    # 真实标签
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap='tab10', alpha=0.6)
    ax2.set_title('True Labels Distribution', fontsize=14)
    ax2.set_xlabel(f'First Principal Component\nExplained Variance: {pca.explained_variance_ratio_[0]:.1%}')
    ax2.set_ylabel(f'Second Principal Component\nExplained Variance: {pca.explained_variance_ratio_[1]:.1%}')
    plt.colorbar(scatter2, ax=ax2, label='True Digit')
    
    # 添加说明文本
    plt.figtext(0.5, 0.02, 
                'PCA reduces the 784-dimensional data to 2D for visualization.\n' +
                'Similar colors in left plot should match with right plot for good clustering.',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('pca_clusters.png', dpi=300)

def plot_sample_images(X, labels, y, n_samples_per_cluster=5):
    """绘制增强版样本图像展示"""
    fig, axes = plt.subplots(10, n_samples_per_cluster, figsize=(15, 25))
    
    for cluster in range(10):
        cluster_samples_idx = np.where(labels == cluster)[0]
        cluster_samples = X[cluster_samples_idx]
        cluster_labels = y[cluster_samples_idx]
        
        # 选择前n_samples_per_cluster个样本
        selected_samples = cluster_samples[:n_samples_per_cluster]
        selected_labels = cluster_labels[:n_samples_per_cluster]
        
        for idx, (sample, true_label) in enumerate(zip(selected_samples, selected_labels)):
            axes[cluster, idx].imshow(sample.reshape(28, 28), cmap='gray')
            axes[cluster, idx].axis('off')
            axes[cluster, idx].set_title(f'True: {true_label}', fontsize=8)
            
        # 添加簇信息
        cluster_label_dist = np.unique(cluster_labels, return_counts=True)
        most_common_digit = cluster_label_dist[0][np.argmax(cluster_label_dist[1])]
        most_common_percent = np.max(cluster_label_dist[1]) / len(cluster_labels) * 100
        
        axes[cluster, 0].set_ylabel(f'Cluster {cluster}\nMost common: {most_common_digit}\n({most_common_percent:.1f}%)',
                                  fontsize=10)
    
    plt.suptitle('Sample Images from Each Cluster\nwith True Labels and Cluster Statistics', 
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300)

def main():
    with timer.timer("Total Execution"):
        # 加载数据
        X, y = load_mnist(n_samples=5000)
        
        # 训练模型
        kmeans = KMeans(n_clusters=10, max_iters=100)
        
        with timer.timer("Model Training"):
            labels = kmeans.fit(X)
        
        # 计算评估指标
        with timer.timer("Metrics Calculation"):
            silhouette_avg = silhouette_score(X, labels)
            cluster_sizes = [np.sum(labels == i) for i in range(10)]
            
            # 计算每个簇的纯度
            cluster_purities = []
            for k in range(10):
                cluster_labels = y[labels == k]
                unique, counts = np.unique(cluster_labels, return_counts=True)
                purity = (np.max(counts) / len(cluster_labels)) * 100
                cluster_purities.append(purity)
        
        # 可视化结果
        with timer.timer("Visualization"):
            print("\n1. Visualizing learned cluster centers with statistics...")
            with timer.timer("Centroids Visualization"):
                visualize_centroids(kmeans.centroids, cluster_sizes, cluster_purities)
            
            print("\n2. Plotting convergence curve and iteration times...")
            with timer.timer("Loss Curve Visualization"):
                plot_loss_curve(kmeans.inertia_history_, kmeans.iteration_times_)
            
            print("\n3. Visualizing cluster distribution with cluster sizes...")
            with timer.timer("Distribution Visualization"):
                plot_cluster_distribution(labels, y, cluster_sizes)
            
            print("\n4. Visualizing clusters in 2D using PCA with silhouette score...")
            with timer.timer("PCA Visualization"):
                visualize_pca_clusters(X, labels, y, silhouette_avg)
            
            print("\n5. Showing sample images from each cluster with true labels...")
            with timer.timer("Sample Images Visualization"):
                plot_sample_images(X, labels, y)
        
        # 打印详细的聚类评估报告
        with timer.timer("Results Report"):
            print("\nClustering Evaluation Report:")
            print(f"Number of iterations to converge: {kmeans.n_iters_}")
            print(f"Silhouette Score: {silhouette_avg:.3f}")
            print("\nCluster Statistics:")
            
            for k in range(10):
                cluster_labels = y[labels == k]
                print(f"\nCluster {k}:")
                print(f"Size: {len(cluster_labels)} samples ({len(cluster_labels)/len(y)*100:.1f}% of total)")
                print(f"Purity: {cluster_purities[k]:.1f}%")
                unique, counts = np.unique(cluster_labels, return_counts=True)
                for label, count in zip(unique, counts):
                    print(f"  Digit {label}: {count} ({count/len(cluster_labels)*100:.1f}%)")
        
        # 打印时间报告
        print(timer.get_report())

if __name__ == "__main__":
    main()