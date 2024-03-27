import torch
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import os

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 圖像轉換
# 定義訓練數據的轉換（包含數據增強）
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加載模型
model = models.efficientnet_b0(weights=None)  # 假設使用 EfficientNetB0
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 假設是二元分類
model.load_state_dict(torch.load('best_model_overall_race.pth'))  # 替換為模型檔案路徑
model = model.to(device)
model.eval()

# 讀取 CSV 檔案
csv_path = '1026公告_projectB_ans.csv'  # CSV 檔案路徑
df = pd.read_csv(csv_path)

# 圖片檔案路徑
image_folder = 'TOPIC/ProjectB/B_testing'  # 替換為圖片檔案夾路徑

# 預測並更新 CSV 檔案
for index, row in df.iterrows():
    file_name = row['Image_name']
    image_path = os.path.join(image_folder, file_name)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        df.at[index, 'label'] = predicted.item()

df.to_csv('1026公告_projectB_ans.csv', index=False)  # 更新的 CSV 檔案路徑
