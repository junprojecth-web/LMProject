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