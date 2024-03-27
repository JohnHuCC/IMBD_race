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
import copy
import argparse

# 檢查CUDA是否可用
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
print('cuda_available:', cuda_available)

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


# 定義不同的模型
models_dict = {
    "ShuffleNetV2": models.shufflenet_v2_x1_0(weights=None),
    "EfficientNetB0": models.efficientnet_b0(weights=None),
    "MobileNetV2": models.mobilenet_v2(weights=None),
    "ResNet-18": models.resnet18(weights=None),
}

# 為不同模型設置不同的 epoch 數
epochs_dict = {
    "ShuffleNetV2": 100,
    "EfficientNetB0": 100,
    "MobileNetV2": 100,
    "ResNet-18": 100,
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
    for images, labels in tqdm(train_loader, desc="Training"):  # 使用tqdm包装
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
kf = KFold(n_splits=5, shuffle=True, random_state=42)
optimizers = {
    # 'Adafactor': Adafactor,
    'Adam': optim.Adam,
    'SGD': optim.SGD
}
# learning_rates = [1e-3, 5e-4, 1e-4]

# script parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='Optimizer: Adam, SGD, etc.')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
args = parser.parse_args()


def initialize_model(model_name, device):
    model = models_dict[model_name]
    # 重新定義全連接層
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

    model.to(device)
    return model


def initialize_optimizer(model, opt_name, lr):
    if opt_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)


def initialize_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, cooldown=2, min_lr=1e-6)


def run_fold(model, optimizer, scheduler, train_idx, val_idx, epochs, writer, fold, device):
    criterion = nn.CrossEntropyLoss()
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        data_full, batch_size=32, sampler=train_subsampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(
        data_full, batch_size=1, sampler=val_subsampler, num_workers=1, pin_memory=True)

    fold_best_accuracy = 0.0
    fold_best_model_state = None

    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(
            model, train_loader, optimizer, criterion)
        val_loss, val_accuracy, val_labels, val_probs = validate_model(
            model, val_loader, criterion)

        # scheduler.step(val_loss)
        current_lr = get_lr(optimizer)

        # 更新當前 fold 的最佳模型
        if val_accuracy > fold_best_accuracy:
            fold_best_accuracy = val_accuracy
            fold_best_model_state = copy.deepcopy(model.state_dict())

        tensorboard_title = f'{model.__class__.__name__}_{optimizer.__class__.__name__}_LR{current_lr}/Fold_{fold+1}_no_scheduler'
        writer.add_scalar(f'{tensorboard_title}/Loss/Train', train_loss, epoch)
        writer.add_scalar(
            f'{tensorboard_title}/Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar(
            f'{tensorboard_title}/Loss/Validation', val_loss, epoch)
        writer.add_scalar(
            f'{tensorboard_title}/Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar(
            f'{tensorboard_title}/Learning Rate', current_lr, epoch)

        # ROC 和 AUC 計算
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        roc_auc = auc(fpr, tpr)
        writer.add_scalar(
            f'{tensorboard_title}/AUC/Validation', roc_auc, epoch)

        # 繪製 ROC 曲線
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        writer.add_figure(
            f'{tensorboard_title}/ROC Curve/Validation', fig, epoch)
        plt.close(fig)

    return fold_best_accuracy, fold_best_model_state, val_accuracy


def run_training(args, device):
    # 加載模型狀態
    shufflenet = initialize_model("ShuffleNetV2", device)
    mobilenet = initialize_model("MobileNetV2",  device)
    efficientnet = initialize_model("EfficientNetB0",  device)

    # 創建一次驗證數據加載器
    _, val_idx = next(iter(kf.split(data_full)))  # 只獲取一次分割
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
    val_loader = DataLoader(
        data_full, batch_size=1, sampler=val_subsampler, num_workers=1, pin_memory=True)

    # 加載並評估堆疊模型的準確度
    stacked_fold_accuracies = []
    for fold in range(5):  # 假設有 5 個折疊

        print(f'fold: {fold+1}')
        pre_shufflenet = torch.load(
            f'best_ShuffleNetV2_fold_{fold + 1}.pth')['state_dict']

        shufflenet.load_state_dict(pre_shufflenet)
        pre_mobilenet = torch.load(
            f'best_MobileNetV2_fold_{fold + 1}.pth')['state_dict']
        mobilenet.load_state_dict(pre_mobilenet)
        pre_efficientnet = torch.load(
            f'best_EfficientNetB0_fold_{fold + 1}.pth')['state_dict']
        efficientnet.load_state_dict(pre_efficientnet)

        # 在當前折疊中評估堆疊模型
        stacked_accuracy = 0
        total_samples = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = predict_with_models(
                images, [shufflenet, mobilenet, efficientnet])
            final_prediction = majority_vote(predictions)
            stacked_accuracy += (final_prediction == labels).sum().item()
            total_samples += labels.size(0)

        stacked_accuracy /= total_samples
        stacked_fold_accuracies.append(stacked_accuracy)

    # # 假設您已有單一模型的準確度數據（需要您自行提供）
    # single_model_fold_accuracies = [...]  # 替換為您的數據

    # 比較和分析每個折疊的單一模型與堆疊模型準確度
    for fold in range(5):
        print(
            f"Fold {fold + 1}:  Stacked Model Accuracy = {stacked_fold_accuracies[fold]}")


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def predict_with_models(image, models):
    # image 已預處理並準備好用於預測
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            prediction = model(image)
            predictions.append(prediction)
    return predictions


def majority_vote(predictions):
    # 假設二元分類
    votes = [torch.argmax(pred, dim=1) for pred in predictions]
    majority_vote = torch.mode(torch.stack(votes, dim=0), dim=0).values
    return majority_vote


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer: Adam, SGD, etc.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    args = parser.parse_args()

    run_training(args, device)
