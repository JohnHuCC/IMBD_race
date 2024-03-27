import os
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
import pytorch_ssim

# 檢查是否有可用的 GPU，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images_from_folder(folder, size=(384, 384)):
    images = {}
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                img = img.convert('L')  # 轉換為灰階
                img = img.resize(size, Image.ANTIALIAS)  # 調整圖片尺寸
                img_tensor = TF.to_tensor(img).unsqueeze(
                    0).to(device)  # 轉換為張量並移至 GPU
                images[filename] = img_tensor
    return images


def filter_images(images, threshold=0.95):
    keys = list(images.keys())
    ssim_module = pytorch_ssim.SSIM(
        window_size=11, size_average=True).to(device)
    to_delete = set()

    for i in tqdm(range(len(keys)), desc="Comparing images"):
        for j in range(i+1, len(keys)):
            img1 = images[keys[i]]
            img2 = images[keys[j]]

            s = ssim_module(img1, img2)

            if s.item() > threshold:
                to_delete.add(keys[j])

    return [k for k in keys if k not in to_delete]


def main():
    folder_path = 'TOPIC/ProjectB/B_traing1/not reworkable'
    images = load_images_from_folder(folder_path)
    filtered_images = filter_images(images)
    print("Filtered Images:", filtered_images)


if __name__ == '__main__':
    main()
