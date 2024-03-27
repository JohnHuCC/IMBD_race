import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('cuda:', device)


class LabelEmbedder(nn.Module):
    def __init__(self, num_labels, embedding_dim):
        super(LabelEmbedder, self).__init__()
        self.embedder = nn.Embedding(num_labels, embedding_dim)

    def forward(self, labels):
        return self.embedder(labels)


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100 + 10, 512),  # 增加 10 是標籤嵌入的大小
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, 384 * 384 * 3),  # 乘以 3 因为是 RGB
            nn.Tanh()
        )
        self.label_embedder = LabelEmbedder(2, 10)  # 假设有两个类别

    def forward(self, noise, labels):
        label_embeddings = self.label_embedder(labels).squeeze(1)
        combined_input = torch.cat((noise, label_embeddings), 1)
        return self.main(combined_input).view(-1, 3, 384, 384)


class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(384 * 384 * 3 + 10, 2048),  # 增加 10 是标签嵌入的大小
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embeddings = self.label_embedder(labels).squeeze(1)
        images_flattened = images.view(images.size(0), -1)
        combined_input = torch.cat((images_flattened, label_embeddings), 1)
        return self.main(combined_input)


generator = ConditionalGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)

optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(),
                        lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# 定義訓練數據的轉換（包含數據增強）
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomVerticalFlip(),    # 隨機垂直翻轉
    transforms.RandomRotation(10),      # 隨機旋轉
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2),  # 顏色抖動
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# 加載數據
train_data = ImageFolder(
    root='TOPIC/ProjectB/B_traing1', transform=transform)
train_loader = DataLoader(
    train_data, batch_size=32,  num_workers=8, pin_memory=True)

epochs = 5
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        # 真实数据
        real_images = images.to(device)
        real_labels = torch.ones(images.size(0), 1).to(device)

        # 假数据标签 - 随机生成
        fake_labels = torch.randint(0, 2, (images.size(0),)).to(device)
        # 假数据
        noise = torch.randn(images.size(0), 100).to(device)
        fake_images = generator(noise, fake_labels)  # 使用噪声和假标签

        # 训练判别器
        discriminator.zero_grad()
        outputs_real = discriminator(real_images, labels)  # 真实图像和真实标签
        real_loss = criterion(outputs_real, real_labels)

        outputs_fake = discriminator(fake_images, fake_labels)  # 假图像和假标签
        fake_loss = criterion(outputs_fake, torch.zeros(
            images.size(0), 1).to(device))
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizerD.step()

        # 训练生成器
        generator.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

        if (i+1) % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')


def generate_and_save_images(generator, num_images, labels, save_dir):
    noise = torch.randn(num_images, 100).to(device)
    fake_images = generator(noise, labels).cpu().detach()

    for i in range(num_images):
        # 将图像数据转换回可保存的格式
        img = (fake_images[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img)

        # 根据标签决定保存路径
        label = labels[i].item()
        folder = 'reworkable' if label == 0 else 'not_reworkable'
        img.save(f'{save_dir}/{folder}/generated_img_{i}.png')


if __name__ == '__main__':
    num_images_to_generate = 2359
    generate_and_save_images(generator, num_images_to_generate, fake_labels,
                             '/Users/johnnyhu/Desktop/THU_bigdata_final/TOPIC/ProjectB/B_traing2')
