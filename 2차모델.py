Python
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

# 1. 설정 및 하이퍼파라미터
DATA_DIR = "./plant-pathology-2021-fgvc8"  # 데이터셋 경로 (압축 해제된 폴더 위치)
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset 정의 (라벨 파싱 및 원핫/인덱스 변환)
class FGVC8Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 고유한 질병 라벨 목록 추출 및 정수 매핑
        self.classes = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label_str = row['labels']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label_str]
        return image, torch.tensor(label_idx, dtype=torch.long)

# 3. 데이터 전처리 (Transform)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 데이터로더 준비
dataset = FGVC8Dataset(CSV_PATH, IMAGE_DIR, transform=train_transform)
num_classes = len(dataset.classes)
print(f"[*] 감지된 세부 질병 클래스 수: {num_classes}개")
print(f"[*] 클래스 목록: {dataset.classes}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 5. 모델 정의 (EfficientNet-B3 기반 전이학습 모델)
class FGVC8TransferModel(nn.Module):
    def __init__(self, num_classes):
        super(FGVC8TransferModel, self).__init__()
        # 사전 학습된 EfficientNet-B3 로드
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        num_features = self.base_model.num_features
        self.feature_extractor = self.base_model.forward_features
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # 세부 질병 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.global_pool(features)
        features = self.flatten(features)
        out = self.classifier(features)
        return out

model = FGVC8TransferModel(num_classes).to(DEVICE)

# 6. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 7. 전이학습 루프 실행
print("[*] FGVC8 전이학습을 시작합니다...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

# 8. 학습된 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': dataset.classes
}, "plant_pathology_fgvc8_model.pth")

print("[*] 전이학습 완료 및 모델 저장 성공: plant_pathology_fgvc8_model.pth")

# ==========================================
# 1. 2차 모델 학습 결과 시각화
# ==========================================

import matplotlib.pyplot as plt

# 학습 결과
epochs = range(1, 9)

train_losses = [
    0.1606,
    0.0732,
    0.0559,
    0.0446,
    0.0372,
    0.0306,
    0.0188,
    0.0153
]

val_losses = [
    0.0886,
    0.0699,
    0.0765,
    0.0556,
    0.0670,
    0.0888,
    0.0779,
    0.0809
]

train_accs = [
    93.75,
    97.23,
    98.02,
    98.36,
    98.62,
    98.77,
    99.18,
    99.36
]

val_accs = [
    95.86,
    97.59,
    98.21,
    98.63,
    96.55,
    98.61,
    98.92,
    98.94
]


# ==========================================
# Loss 그래프
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.xticks(epochs)
plt.legend()
plt.grid(True)

plt.show()


# ==========================================
# Accuracy 그래프
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    epochs,
    train_accs,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accs,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")

plt.xticks(epochs)
plt.legend()
plt.grid(True)

plt.show()


import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

# 1. 설정 및 하이퍼파라미터
DATA_DIR = "./plant-pathology-2021-fgvc8"  # 데이터셋 경로 (압축 해제된 폴더 위치)
CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset 정의 (라벨 파싱 및 원핫/인덱스 변환)
class FGVC8Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 고유한 질병 라벨 목록 추출 및 정수 매핑
        self.classes = sorted(self.df['labels'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label_str = row['labels']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.class_to_idx[label_str]
        return image, torch.tensor(label_idx, dtype=torch.long)

# 3. 데이터 전처리 (Transform)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. 데이터로더 준비
dataset = FGVC8Dataset(CSV_PATH, IMAGE_DIR, transform=train_transform)
num_classes = len(dataset.classes)
print(f"[*] 감지된 세부 질병 클래스 수: {num_classes}개")
print(f"[*] 클래스 목록: {dataset.classes}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 5. 모델 정의 (EfficientNet-B3 기반 전이학습 모델)
class FGVC8TransferModel(nn.Module):
    def __init__(self, num_classes):
        super(FGVC8TransferModel, self).__init__()
        # 사전 학습된 EfficientNet-B3 로드
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        num_features = self.base_model.num_features
        self.feature_extractor = self.base_model.forward_features
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # 세부 질병 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.global_pool(features)
        features = self.flatten(features)
        out = self.classifier(features)
        return out

model = FGVC8TransferModel(num_classes).to(DEVICE)

# 6. 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 7. 전이학습 루프 실행
print("[*] FGVC8 전이학습을 시작합니다...")
model.train()

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for images, labels in progress_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

# 8. 학습된 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': dataset.classes
}, "plant_pathology_fgvc8_model.pth")

print("[*] 전이학습 완료 및 모델 저장 성공: plant_pathology_fgvc8_model.pth")