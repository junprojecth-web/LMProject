## 해당 모델의 학습과 시각화는 Kaggle 에서 진행. Kaggle notebook "DLProject1"
### 아래 코드는 저장용##
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import timm

# ============================================================
# 1. 설정 및 디바이스 지정 (Kaggle 경로 반영)
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 300  # EfficientNet-B3 권장 해상도
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Kaggle 입력 경로 설정 (추가한 데이터셋 폴더명에 맞춤)
# 예: /kaggle/input/weitianqi 또는 사용 중인 폴더명
DATASET_DIR = Path("/kaggle/input") 

# 이미지 확장자 정의
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

print("Kaggle 데이터셋 경로 스캔 중...")
image_records = []

# /kaggle/input 하위의 모든 파일을 재귀적으로 탐색하여 이미지 수집
if DATASET_DIR.exists():
    for file_path in DATASET_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_records.append({
                "image_path": str(file_path),
                "parent_folder": file_path.parent.name
            })

df = pd.DataFrame(image_records)
print(f"학습에 사용할 총 이미지 수: {len(df):,}장")

if len(df) == 0:
    raise ValueError("수집된 이미지가 없습니다. Kaggle Input에 데이터셋이 올바르게 추가되었는지 확인해주세요.")


# ============================================================
# 2. 임시 라벨링 함수 (폴더명 기반 자동 라벨링)
# ============================================================
def assign_state_label(folder_name):
    folder_lower = str(folder_name).lower()
    if "healthy" in folder_lower or "normal" in folder_lower:
        return 0  # 정상
    elif "disease" in folder_lower or "rot" in folder_lower or "blight" in folder_lower or "spot" in folder_lower:
        return 1  # 병충해
    elif "yellow" in folder_lower or "curl" in folder_lower or "wilt" in folder_lower or "stress" in folder_lower:
        return 2  # 잎 이상
    else:
        return 3  # 기타

df["label"] = df["parent_folder"].apply(assign_state_label)

CLASSES = {0: "정상", 1: "병충해", 2: "잎 이상", 3: "기타"}
print("\n[상태 라벨별 분포]")
print(df["label"].map(CLASSES).value_counts())


# ============================================================
# 3. 데이터셋 및 데이터로더 정의
# ============================================================
class PlantDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = row["label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터가 너무 많을 경우 테스트를 위해 일부만 쓰려면 df 대신 df.sample(n=5000) 등 사용 가능
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_dataset = PlantDataset(train_df, transform=train_transform)
val_dataset = PlantDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# 4. 모델 생성 (EfficientNet-B3)
# ============================================================
print("\nEfficientNet-B3 모델 로딩 중...")
# 1. 모델 구조 생성
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=4)
model.to(DEVICE)

# 2. ★저장해 둔 가중치 파일이 있다면 불러오기★
checkpoint_path = 'model/best_plant_model.pth'  # 또는 중간 체크포인트 경로
if os.path.exists(checkpoint_path):
  model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
  print(
      f"기존에 학습된 가중치({checkpoint_path})를 성공적으로 불러왔습니다! 이어서"
      " 학습을 시작합니다."
  )
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)


# 이후 기존과 동일하게 optimizer, criterion 설정 후 학습 루프 실행
# ============================================================
# 5. 학습 루프 (Training Loop)
# ============================================================
print("\n" + "="*50)
print("모델 학습 시작")
print("="*50)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_loss = running_loss / total_train
    train_acc = correct_train / total_train

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    val_loss = val_loss / total_val
    val_acc = correct_val / total_val
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/best_plant_model.pth")
        print(" -> 최적 모델 저장 완료 (model/best_plant_model.pth)")

print("\n모든 학습이 완료되었습니다!")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 학습 중 중간 저장
# ============================================================
import os
import torch

os.makedirs("dlmodel01", exist_ok=True)
torch.save(model.state_dict(), "model/checkpoint_epoch_saved.pth")
print("중간 가중치 저장 완료!")

# ============================================================
# 모델 다운로드 
# ============================================================
from IPython.display import FileLink
FileLink(r'model/best_plant_model.pth')


# ============================================================
# 검증 셋을 이용한 모델 평가 및 시각화
# ============================================================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 1. 클래스별 상세 리포트 출력 (Precision, Recall, F1-score)
class_names = ["정상", "병충해", "잎 이상", "기타"]
print("\n[Classification Report]")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 2. 혼동 행렬(Confusion Matrix) 시각화
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.title("Plant State Confusion Matrix", fontsize=14, fontweight='bold')
plt.show()

