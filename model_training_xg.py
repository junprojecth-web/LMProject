# 전처리한 데이터를 분석하여 작물을 추천하는 모델은 DL 모델이 아닌, ML 모델을 사용.
# 이전 ML 모델 비교하여 최종 서비스용으로 XGBoost 모델을 학습 시켜 적용.
# 국내 재배 환경에서 기후 변수 간 비선형 관계를 학습하고, 특히 최저기온 및 한파 안전 마진과 같은 파생변수를 효과적으로 반영할 수 있어 XGBoost를 최종 모델로 선정 함

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# 0. 기본 설정
# ============================================================

print("=" * 60)
print("기후 적응형 작물·과수 추천 XGBoost 최종 모델 학습")
print("=" * 60)

os.makedirs("model", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# 데이터 로드
df = pd.read_csv(
    "./dataset/processed_ml_dataset.csv",
    encoding="utf-8-sig"
)

print(f"데이터 크기: {df.shape}")
print(f"전체 지역 수: {df['region'].nunique()}")
print(f"전체 작물/과수 수: {df['crop_name'].nunique()}")

# ============================================================
# 1. 컬럼명 정리
# ============================================================

rename_map = {
    "작물명": "crop_name",
    "생육적온_최저(℃)": "opt_temp_min",
    "생육적온_최고(℃)": "opt_temp_max",
    "한계생육온도(℃)": "frost_limit_temp",
    "적정습도(%)": "opt_humidity",
    "土壤pH_최저": "soil_ph_min",
    "토양pH_최저": "soil_ph_min",
    "토양pH_최고": "soil_ph_max",
    "수익성(1-5)": "profit_score"
}

df = df.rename(columns=rename_map)

# ============================================================
# 2. 파생변수 생성
# ============================================================

# 생육적온 평균
if "opt_temp_avg" not in df.columns:
    df["opt_temp_avg"] = (
        df["opt_temp_min"] + df["opt_temp_max"]
    ) / 2

# 실제 평균기온과 생육적온 중심의 차이
if "temp_diff_from_opt" not in df.columns:
    df["temp_diff_from_opt"] = (
        df["avg_temp"] - df["opt_temp_avg"]
    ).abs()

# 최저기온 대비 한계생육온도 안전마진
if "frost_safety_margin" not in df.columns:
    df["frost_safety_margin"] = (
        df["min_temp"] - df["frost_limit_temp"]
    )

# ============================================================
# 3. 사용할 Feature 정의
# ============================================================

feature_cols = [
    "min_temp",
    "max_temp",
    "avg_temp",
    "avg_rhm",
    "annual_rn",

    "opt_temp_min",
    "opt_temp_max",
    "frost_limit_temp",
    "opt_humidity",

    "soil_ph_min",
    "soil_ph_max",

    "opt_temp_avg",
    "temp_diff_from_opt",
    "frost_safety_margin"
]

# 실제 존재하는 컬럼만 사용
feature_cols = [
    col for col in feature_cols
    if col in df.columns
]

print("\n사용 Feature")
for col in feature_cols:
    print(" -", col)

# ============================================================
# 4. X / y 생성
# ============================================================

X = df[feature_cols].copy()

# 목표변수
# 0 = 부적합
# 1 = 적합
y = df["suitability"].astype(int)

print("\n타깃 분포")
print(y.value_counts())
print(y.value_counts(normalize=True))

# ============================================================
# 5. 클래스 불균형 처리
# ============================================================

negative = (y == 0).sum()
positive = (y == 1).sum()

scale_pos_weight = negative / positive

print("\n클래스 불균형 보정값")
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# ============================================================
# 6. 최종 XGBoost 모델 생성
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,
    n_jobs=-1
)

# ============================================================
# 7. 5-Fold Cross Validation
# ============================================================

print("\n" + "=" * 60)
print("5-Fold Cross Validation")
print("=" * 60)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

f1_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)

roc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(
    f"XGBoost 5-Fold F1 : "
    f"{f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
)

print(
    f"XGBoost 5-Fold ROC-AUC : "
    f"{roc_scores.mean():.4f} ± {roc_scores.std():.4f}"
)

# ============================================================
# 8. 전체 데이터로 최종 모델 학습
# ============================================================

print("\n" + "=" * 60)
print("전체 데이터 기반 최종 XGBoost 학습")
print("=" * 60)

model.fit(X, y)

print("✔ 최종 모델 학습 완료")

# ============================================================
# 9. 전체 데이터 기준 예측
# ============================================================

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

print("\n최종 모델 학습 데이터 평가")
print(f"Accuracy  : {accuracy_score(y, pred):.4f}")
print(f"Precision : {precision_score(y, pred, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y, pred, zero_division=0):.4f}")
print(f"F1-Score  : {f1_score(y, pred, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y, proba):.4f}")

# ============================================================
# 10. Feature Importance
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))

importance.plot(kind="barh")

plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig(
    "figs/xgboost_feature_importance.png",
    dpi=150
)

plt.close()

print("✔ Feature Importance 저장 완료")

# ============================================================
# 11. Confusion Matrix
# ============================================================

cm = confusion_matrix(y, pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["부적합", "적합"]
)

fig, ax = plt.subplots(figsize=(5, 5))

disp.plot(ax=ax)

plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "figs/xgboost_confusion_matrix.png",
    dpi=150
)

plt.close()

print("✔ Confusion Matrix 저장 완료")

# ============================================================
# 12. 모델 저장
# ============================================================

model.save_model(
    "model/xgboost_crop_recommendation.json"
)

# Feature 정보 저장
pd.DataFrame({
    "feature": feature_cols
}).to_csv(
    "model/feature_columns.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n✔ 모델 저장 완료")
print("   model/xgboost_crop_recommendation.json")
print("   model/feature_columns.csv")

# ============================================================
# 13. 작물별 학습 데이터 예측 결과 저장
# ============================================================

result_df = df[
    ["region", "crop_name"]
].copy()

result_df["predicted_probability"] = proba
result_df["predicted_suitability"] = pred

result_df.to_csv(
    "model/xgboost_training_prediction.csv",
    index=False,
    encoding="utf-8-sig"
)

print("✔ 예측 결과 저장 완료")

print("\n" + "=" * 60)
print("XGBoost 최종 모델 학습 종료")
print("=" * 60)