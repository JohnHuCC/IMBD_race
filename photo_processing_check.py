import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageEnhance, ImageFilter

# 假设我们有一个PIL图像：image
image_path = '/Users/johnnyhu/Desktop/THU_bigdata_final/TOPIC/ProjectB/B_traing1/not reworkable/FBAFC0049-03.jpg'
image = Image.open(image_path)


class RandomSharpness(object):
    def __init__(self, probability=0.5, sharpness_factor=2.0):
        self.probability = probability
        self.sharpness_factor = sharpness_factor

    def __call__(self, img):
        if random.random() < self.probability:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.sharpness_factor)
        return img


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # 检查 img 的类型并相应地获取尺寸
        if isinstance(img, torch.Tensor):
            h = img.size(1)
            w = img.size(2)
        elif isinstance(img, Image.Image):
            w, h = img.size

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., probability=0.5):
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


# 定义转换
train_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomVerticalFlip(),    # 隨機垂直翻轉
    transforms.RandomRotation(10),      # 隨機旋轉
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2),  # 顏色抖動
    transforms.Lambda(RandomSharpness(probability=0.5, sharpness_factor=2.0)),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.05, probability=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 应用转换
transformed_image = train_transform(image)

# 转换为 NumPy 数组用于显示


def imshow(inp, title=None):
    """Imshow for Tensor."""
    if isinstance(inp, torch.Tensor):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = inp * std + mean  # 反归一化，先乘以std再加上mean
        inp = np.clip(inp, 0, 1)  # 确保值在0到1之间
        plt.imshow(inp)
    else:  # 如果输入是 PIL.Image.Image，则直接显示
        plt.imshow(inp)
    if title is not None:
        plt.title(title)


# 显示原始图像和转换后的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
imshow(transformed_image)  # 直接传入 PIL.Image.Image 对象
plt.title('Transformed Image')
plt.show()
