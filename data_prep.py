import argparse
from os.path import join
from pathlib import Path

import cv2
import numpy as np

def nparray_to_str(X):
    """
    Chuyển đổi mảng numpy thành chuỗi để lưu vào tệp.
    """
    return '\n'.join(' '.join(map(str, row)) for row in X)

def kmeans_plus_plus(data, k):
    """
    Thuật toán KMeans++ để chọn tâm cụm ban đầu.
    """
    centroids = [data[np.random.choice(range(len(data)))]]

    for _ in range(1, k):
        D2 = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in data])
        probabilities = D2 / D2.sum() if D2.sum() > 0 else np.ones_like(D2) / len(D2)
        cumulative_probs = probabilities.cumsum()
        next_centroid = data[np.searchsorted(cumulative_probs, np.random.rand())]
        centroids.append(next_centroid)

    return np.array(centroids)

def kmeans(data, initial_centroids, max_iter=30):
    """
    Thuật toán phân cụm KMeans.
    """
    k = len(initial_centroids)
    centers = initial_centroids.copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([data[labels == j].mean(axis=0) if len(data[labels == j]) > 0 else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers):
            break 

        centers = new_centers

    return labels, centers

def main(src_img, dst_folder, k):
    """
    Xử lý chính: đọc ảnh, phân cụm và lưu kết quả.
    """
    Path(dst_folder).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(src_img)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {src_img}")

    br, kl, _ = img.shape
    data = img.reshape((-1, 3)).astype(np.float32) / 255.0

    points_path = join(dst_folder, 'points.txt')
    np.savetxt(points_path, data, fmt='%f')
    print(f'Danh sách điểm được lưu tại: {points_path}')

    initial_centroids = kmeans_plus_plus(data, k)
    labels, centers = kmeans(data, initial_centroids)

    labels_img = labels.reshape((br, kl))
    centers = (centers * 255).astype(np.uint8)
    segmented_img = centers[labels_img]

    segmented_img_path = join(dst_folder, 'segmented_image.png')
    cv2.imwrite(segmented_img_path, segmented_img)
    print(f'Ảnh phân đoạn được lưu tại: {segmented_img_path}')

    clusters_path = join(dst_folder, 'clusters.txt')
    cluster_data = np.hstack((np.arange(1, k + 1).reshape((-1, 1)), centers))
    np.savetxt(clusters_path, cluster_data, fmt='%d %d %d %d')
    print(f'Tâm cụm được lưu tại: {clusters_path}')

if __name__ == '__main__':
    # Tạo đối tượng parser để nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser(description='Script tạo tệp points.txt và clusters.txt từ ảnh đầu vào.')
    parser.add_argument('--src_img', type=str, required=True, help='Đường dẫn tới ảnh nguồn.')
    parser.add_argument('--dst_folder', type=str, required=True, help='Thư mục lưu kết quả.')
    parser.add_argument('--k_init_centroids', type=int, default=5, help='Số tâm cụm ban đầu (mặc định: 5).')

    args = parser.parse_args()
    main(args.src_img, args.dst_folder, args.k_init_centroids)
