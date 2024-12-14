import argparse
from glob import glob
from os.path import join, isdir, isfile

import cv2
import numpy as np
from PIL import Image

# Tạo parser để phân tích các tham số đầu vào từ dòng lệnh
parser = argparse.ArgumentParser(description='Script này hiển thị các cụm được ước lượng.')

parser.add_argument('--clusters_path', type=str, help='Đường dẫn tới tệp hoặc thư mục chứa các cụm đã tạo.')
parser.add_argument('--src_img', type=str, help='Đường dẫn tới ảnh nguồn.')
parser.add_argument('--dst_img', type=str, help='Đường dẫn tới ảnh sẽ được lưu sau xử lý.')

args = parser.parse_args()

# Hàm tải dữ liệu cụm từ tệp hoặc thư mục
# Đọc và hợp nhất các tệp cụm nếu là thư mục

def load_clusters(path):
    if isdir(path):
        files = glob(join(path, 'part-r-*[0-9]'))
    elif isfile(path): 
        files = [path]
    else:
        raise Exception('Đường dẫn không hợp lệ.')

    centroids = [load_nparray(file)[:, 1:] for file in files]
    centroids = np.concatenate(centroids, axis=0).reshape(-1, centroids[0].shape[-1])
    return centroids

# Hàm đọc dữ liệu từ tệp và chuyển đổi thành mảng numpy

def load_nparray(file):
    data = []
    with open(file) as f:
        for line in f:
            data.append(np.array([float(num) for num in line.split(' ')]))

    return np.stack(data).astype(float)

# Hàm chuyển đổi ảnh thành ảnh nhị phân dựa trên ngưỡng
# Các pixel nhỏ hơn ngưỡng sẽ thành 0, lớn hơn hoặc bằng thì thành 255

def binary_image(kmeans_img_path, threshold_value):
    GRAYSCALE_IMAGE = Image.open(kmeans_img_path).convert('L')
    GRAYSCALE_PIXEL = GRAYSCALE_IMAGE.load()

    horizontal_size = GRAYSCALE_IMAGE.size[0]
    vertical_size = GRAYSCALE_IMAGE.size[1]

    for x in range(horizontal_size):
        for y in range(vertical_size):
            if GRAYSCALE_PIXEL[x, y] < threshold_value:
                GRAYSCALE_PIXEL[x, y] = 0
            else:
                GRAYSCALE_PIXEL[x, y] = 255

    saved_filename = './tmp/binary_image_' + str(threshold_value) + '.jpg'
    GRAYSCALE_IMAGE.save(saved_filename)

# Hàm chính để xử lý các cụm và lưu ảnh đầu ra
def main(clusters_path, kmeans_img, dst_img):
    clusters = load_clusters(clusters_path)

    img = cv2.imread(kmeans_img)
    shape = img.shape

    binary_image(kmeans_img, 100)

    gbr = cv2.imread('./tmp/binary_image_100.jpg')

    # Áp dụng toán tử xói mòn và giãn nở để làm mịn các vùng cụm
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(gbr, kernel, iterations=4)  
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3) 

    cv2.imwrite(dst_img, img_dilation)
    print(f'Ảnh sau xử lý được lưu tại: {dst_img}')

# Điểm bắt đầu của chương trình
if __name__ == '__main__':
    args = parser.parse_args()
    main(args.clusters_path, args.src_img, args.dst_img)
