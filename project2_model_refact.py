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
    "ShuffleNetV2": 60,
    "EfficientNetB0": 60,
    "MobileNetV2": 70,
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
kf = KFold(n_splits=10, shuffle=True, random_state=42)
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

    return fold_best_accuracy, fold_best_model_state


def run_training(args, device):
    best_accuracy = 0.0
    best_model_state = None
    best_params = {}
    fold_best_models = {}  # 用來儲存每個折疊的最佳模型
    fold_accuracies = {}
    print(f"Evaluating with optimizer: {args.optimizer}, lr: {args.lr}")
    writer = SummaryWriter(
        f'runs/{args.model}_{args.optimizer}_LR{args.lr}_no_scheduler')

    for fold, (train_idx, val_idx) in enumerate(kf.split(data_full)):
        print(f'Fold {fold + 1}')
        model = initialize_model(args.model, device)
        optimizer = initialize_optimizer(model, args.optimizer, args.lr)
        scheduler = initialize_scheduler(optimizer)

        # Reset model weights and optimizer state
        model.apply(reset_weights)
        optimizer = initialize_optimizer(model, args.optimizer, args.lr)

        # Run training and validation for the current fold
        fold_accuracy, fold_best_model_state = run_fold(model, optimizer, scheduler, train_idx,
                                                        val_idx, epochs_dict[args.model], writer, fold, device)

        # Save the best model state for the current fold
        fold_best_models[fold] = fold_best_model_state

        # Save the accuracy for the current fold
        fold_accuracies[fold] = fold_accuracy

        # Check and save the best model
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            best_params = {'model': args.model,
                           'optimizer': args.optimizer, 'lr': args.lr}

    writer.close()
    # Save best models of each fold
    for fold, model_state in fold_best_models.items():
        model_accuracy = fold_accuracies[fold]
        torch.save({
            'state_dict': model_state,
            'accuracy': model_accuracy
        }, f'best_{args.model}_fold_{fold+1}.pth')

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model_overall.pth')
        print(f"Best model saved with accuracy: {best_accuracy}")
        print(f"Best parameters: {best_params}")


def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer: Adam, SGD, etc.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    args = parser.parse_args()

    run_training(args, device)
