# train_plant_model.py
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
import timm  # timm 라이브러리 (EfficientNet-B3 활용을 위해 권장)

# ============================================================
# 1. 설정 및 디바이스 지정
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 300  # EfficientNet-B3 권장 해상도
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 앞서 진단 결과로 생성된 all_image_inventory.csv 활용
CSV_PATH = "all_image_inventory.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"'{CSV_PATH}' 파일이 없습니다. 먼저 이미지 진단 스크립트를 실행해주세요.")

df = pd.read_csv(CSV_PATH)

# Kaggle_2 (weitianqi) 데이터만 필터링하거나 전체 활용
df = df[df["dataset"] == "Kaggle_2"].reset_index(drop=True)

print(f"학습에 사용할 총 이미지 수: {len(df):,}장")


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

# 클래스 레이블 매핑
CLASSES = {0: "정상", 1: "병충해", 2: "잎 이상", 3: "기타"}
print("\n[상태 라벨별 분포]")
print(df["label"].map(CLASSES).value_counts())


# ============================================================
# 3. 데이터셋 및 데이터로더 정의
# ============================================================
class PlantDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
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
            # 이미지 손상 시 대체 검은 화면 처리
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 이미지 전처리 및 증강 (Data Augmentation)
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

# Train / Validation 분할 (8:2)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_dataset = PlantDataset(train_df, transform=train_transform)
val_dataset = PlantDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# 4. 모델 생성 (EfficientNet-B3)
# ============================================================
print("\nEfficientNet-B3 모델 로딩 중...")
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=4)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)


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

    # Validation 검증
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

    # 최고 성능 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), "model/best_plant_model.pth")
        print(" -> 최적 모델 저장 완료 (model/best_plant_model.pth)")

print("\n모든 학습이 완료되었습니다!")