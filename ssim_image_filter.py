import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm.contrib.concurrent import process_map  # 用於多進程的 tqdm
from tqdm import tqdm  # 普通的 tqdm 進度條


def load_images_from_folder(folder, size=(384, 384)):
    images = {}
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.convert('L')  # 轉換為灰階
                img = img.resize(size, Image.ANTIALIAS)  # 調整圖片尺寸
                images[filename] = np.array(img)
    return images


def compare_image_pair(pair):
    img1, img2, threshold = pair
    s = ssim(img1, img2)
    return s > threshold


def filter_images(images, threshold=0.95):
    keys = list(images.keys())
    pairs = [(images[keys[i]], images[keys[j]], threshold)
             for i in range(len(keys)) for j in range(i+1, len(keys))]

    # 設置 chunksize 以提高性能
    results = process_map(compare_image_pair, pairs, max_workers=os.cpu_count(
    ), chunksize=1, desc="Comparing images")

    to_delete = set()
    for i, result in enumerate(results):
        if result:
            _, j = divmod(i, len(keys)-1)
            to_delete.add(keys[j+1+i//len(keys)])

    return [k for k in keys if k not in to_delete]


def main():
    folder_path = 'TOPIC/ProjectB/B_traing1/not reworkable'
    images = load_images_from_folder(folder_path)
    filtered_images = filter_images(images)
    print("Filtered Images:", filtered_images)


if __name__ == '__main__':
    main()
