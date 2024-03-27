import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import timm


# 檢查CUDA是否可用
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('cuda_available:', cuda_available)

# 定義訓練數據的轉換（包含數據增強）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.RandomVerticalFlip(),    # 隨機垂直翻轉
    transforms.RandomRotation(10),      # 隨機旋轉
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2),  # 顏色抖動
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 定義驗證數據的轉換
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加載數據
data_full = datasets.ImageFolder(
    root='TOPIC/ProjectB/B_traing1', transform=train_transform)

# # 計算訓練集和驗證集的大小
# total_size = len(data_full)
# train_size = int(0.9 * total_size)
# val_size = total_size - train_size

train_idx = list(range(0, 2250))+list(range(2500, 4750))
val_idx = list(range(2250, 2500))+list(range(4750, 5000))

# 創建訓練集和驗證集
data_train = Subset(data_full, train_idx)
data_val = Subset(data_full, val_idx)

# 定義不同的模型
models_dict = {
    # "ResNet-18": models.resnet18(weights=None),
    "ViT": timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
}


# 重新定義全連接層
for model_name, model in models_dict.items():
    if model_name == "ResNet-18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)


# 訓練函數
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training"):  # 使用tqdm包装
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# 驗證函數


def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):  # 使用tqdm包装
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
    avg_loss = val_loss / len(val_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy, all_labels, all_probs

# 主執行函數


def run_training():
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        writer = SummaryWriter(f'runs/{name}')
        model.to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min')
        criterion = nn.CrossEntropyLoss()

       # 使用全部的訓練集和驗證集
        train_loader = DataLoader(
            data_train, batch_size=128, num_workers=8, pin_memory=True, shuffle=True)
        val_loader = DataLoader(data_val, batch_size=1,
                                num_workers=8, pin_memory=True, shuffle=False)

        for epoch in range(1000):
            train_loss, train_accuracy = train_model(
                model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, val_labels, val_probs = validate_model(
                model, val_loader, criterion)
            scheduler.step(val_loss)
            writer.add_scalar(
                f'Fold_Loss/Train', train_loss, epoch)
            writer.add_scalar(
                f'Fold_Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar(
                f'Fold_Loss/Validation', val_loss, epoch)
            writer.add_scalar(
                f'Fold_Accuracy/Validation', val_accuracy, epoch)

            # # ROC and AUC calculations
            # fpr, tpr, _ = roc_curve(val_labels, val_probs)
            # roc_auc = auc(fpr, tpr)
            # writer.add_scalar(
            #     f'Fold_AUC/Validation', roc_auc, epoch)

            # # Plot ROC curve
            # fig, ax = plt.subplots()
            # ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
            # ax.plot([0, 1], [0, 1], 'k--')
            # ax.set_xlabel('False Positive Rate')
            # ax.set_ylabel('True Positive Rate')
            # ax.set_title('Receiver Operating Characteristic')
            # ax.legend(loc="lower right")
            # writer.add_figure(
            #     f'Fold_ROC Curve/Validation', fig, epoch)
            # plt.close(fig)
        writer.close()


run_training()
