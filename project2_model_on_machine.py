import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Adafactor
import copy
import transformers
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', force=True)

# 檢查CUDA是否可用
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# print('cuda_available:', cuda_available)

# 定義訓練數據的轉換（包含數據增強）
train_transform = transforms.Compose([
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

# 定義驗證數據的轉換
val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加載數據
data_full = datasets.ImageFolder(
    root='TOPIC/ProjectB/B_traing1', transform=train_transform)


train_idx = list(range(0, 2250))+list(range(2500, 4750))
val_idx = list(range(2250, 2500))+list(range(4750, 5000))

# 創建訓練集和驗證集
data_train = Subset(data_full, train_idx)
data_val = Subset(data_full, val_idx)

# 定義不同的模型
models_dict = {
    "ShuffleNetV2": models.shufflenet_v2_x1_0(weights=None),
    "ResNet-18": models.resnet18(weights=None),
    "MobileNetV2": models.mobilenet_v2(weights=None),
    "EfficientNetB0": models.efficientnet_b0(weights=None),
}


# 重新定義全連接層
for model_name, model in models_dict.items():
    if model_name == "MobileNetV2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    elif model_name == "ShuffleNetV2":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
    elif model_name == "EfficientNetB0":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    elif model_name == "ResNet-18":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)


# 訓練函數
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):  # 使用tqdm包装
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
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
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):  # 使用tqdm包装
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
kf = KFold(n_splits=10, shuffle=True, random_state=42)
optimizers = {
    # 'Adafactor': Adafactor,
    'Adam': optim.Adam,
    'SGD': optim.SGD
}
learning_rates = [1e-2, 5e-3, 1e-3]


def run_training():
    best_accuracy = 0.0
    best_model_state = None
    best_params = {}
    with open('training_log.txt', 'w') as log_file:
        for opt_name, opt_func in optimizers.items():
            for lr in learning_rates:
                print(f"Evaluating with optimizer: {opt_name}, lr: {lr}")

                for name, model in models_dict.items():
                    # 重置模型到初始狀態
                    model.apply(reset_weights)
                    model.to(device)
                    print(f"Evaluating {name}...")
                    # 根據優化器類型配置優化器
                    if opt_name == 'SGD':
                        optimizer = opt_func(
                            model.parameters(), lr=lr, momentum=0.9)
                    elif opt_name == 'Adam':
                        optimizer = opt_func(model.parameters(), lr=lr)
                    elif opt_name == 'Adafactor':
                        optimizer = opt_func(
                            model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min')

                    criterion = nn.CrossEntropyLoss()

                    for fold, (train_idx, val_idx) in enumerate(kf.split(data_full)):
                        print(f'Fold {fold + 1}')
                        # 重置模型到初始狀態
                        model.apply(reset_weights)
                        # 創建訓練集和驗證集的子集
                        train_subsampler = torch.utils.data.SubsetRandomSampler(
                            train_idx)
                        val_subsampler = torch.utils.data.SubsetRandomSampler(
                            val_idx)

                        train_loader = DataLoader(
                            data_full, batch_size=64, sampler=train_subsampler, num_workers=8, pin_memory=True)
                        val_loader = DataLoader(
                            data_full, batch_size=1, sampler=val_subsampler, num_workers=1, pin_memory=True)

                        # 訓練和驗證
                        for epoch in range(100):
                            train_loss, train_accuracy = train_model(
                                model, train_loader, optimizer, criterion)
                            # 驗證循環
                            val_loss, val_accuracy, val_labels, val_probs = validate_model(
                                model, val_loader, criterion)

                            # 如果使用學習率調度器
                            scheduler.step(val_loss)

                            # 檢查並保存最佳模型
                            if val_accuracy > best_accuracy:
                                best_accuracy = val_accuracy
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                                best_params = {'model': name,
                                               'optimizer': opt_name, 'lr': lr}
                            tqdm.write(
                                f'Model: {name}, Optimizer: {opt_name}, LR: {lr}, Fold: {fold + 1}, Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
                            log_message = f'Model: {name}, Optimizer: {opt_name}, LR: {lr}, Fold: {fold + 1}, Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n'
                            log_file.write(log_message)
                            log_file.flush()  # 確保信息立即寫入文件
    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')
        print(f"Best model saved with accuracy: {best_accuracy}")
        print(f"Best parameters: {best_params}")

# 函數以重置模型權重


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


if __name__ == '__main__':
    run_training()
