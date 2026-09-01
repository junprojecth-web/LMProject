기후 적응형 열대작물·과수 추천 서비스

기후 데이터와 작물의 생육조건을 머신러닝으로 분석하여 지역별 재배 적합성을 예측하고, 재배 가능한 작물 중 수익성이 높은 작물을 추천하는 서비스입니다.

기후변화로 국내에서 재배 가능한 열대·아열대 작물의 범위가 확대되는 상황에서, 지역별 기후조건과 작물의 생육조건을 데이터 기반으로 비교하여 재배 의사결정을 지원하는 것을 목표로 합니다.

프로젝트 개요
항목	내용
프로젝트명	기후 적응형 열대작물·과수 추천 서비스
프로젝트 유형	머신러닝 기반 추천 서비스
주요 기술	Python, Pandas, Scikit-learn, RandomForest, XGBoost, LightGBM, Streamlit
데이터	국내 98개 지점 기후 데이터 + 열대·과수 100종 생육조건 데이터
데이터 규모	9,800건
예측 대상	작물 재배 적합성
최종 서비스	지역·면적·투자금 기반 작물 추천
개발 환경	VS Code / Jupyter Notebook
프로젝트 배경

기후변화로 국내 평균기온이 상승하면서 애플망고, 파파야, 아보카도와 같은 기존 아열대·열대 지역의 작물도 국내 일부 지역에서 재배 가능성이 확대되고 있습니다.

하지만 수익성이 높은 작물이라고 해서 해당 지역에서 실제로 재배할 수 있는 것은 아닙니다.

따라서 본 프로젝트에서는

"수익성이 높은 작물인가?"보다 먼저 "해당 지역에서 재배가 가능한가?"를 판단

하는 것을 핵심 문제로 정의했습니다.

프로젝트 목표
국내 98개 지역의 기후조건과 100종 작물의 생육조건을 결합
지역별·작물별 재배 적합성 예측 모델 구축
여러 머신러닝 모델의 성능 비교
하이퍼파라미터 튜닝을 통한 모델 고도화
재배 가능한 작물 중 수익성이 높은 작물을 선별
Streamlit을 이용한 사용자 맞춤형 작물 추천 서비스 구현
데이터
기후 데이터

국내 98개 관측지점의 최근 10년 기상 데이터를 활용했습니다.

주요 변수:

평균기온
최고기온
최저기온
평균 상대습도
연강수량
작물 생육조건 데이터

열대작물 및 과수 100종에 대한 데이터를 활용했습니다.

주요 변수:

생육적온 최저/최고
한계생육온도
적정습도
토양 pH
재배난이도
수익성
국내시장성

두 데이터를 조합하여 98개 지역 × 100개 작물 = 총 9,800건의 분석 데이터를 구성했습니다.

데이터 구조

모델의 주요 입력 변수는 총 11개입니다.

[지역 기후조건]
├── 최저기온
├── 최고기온
├── 평균기온
├── 평균습도
└── 연강수량

[작물 생육조건]
├── 생육적온 최저
├── 생육적온 최고
├── 한계생육온도
├── 적정습도
├── 토양 pH 최저
└── 토양 pH 최고

                ↓

        재배 적합성 예측
        suitability
        1 = 적합
        0 = 부적합

데이터 불균형

전체 9,800건 중 실제 재배 적합 사례는 660건(6.7%)으로 클래스 불균형이 존재했습니다.

따라서 단순 Accuracy만으로 모델을 평가하지 않고,

Precision
Recall
F1-score
ROC-AUC
Confusion Matrix

를 함께 사용했습니다.

모델 학습에서는 class_weight='balanced'를 적용하여 소수 클래스의 영향을 높였습니다.

EDA 주요 결과
1. 온도 조건이 가장 중요한 요인

분석 결과 한계생육온도와 지역 최저기온이 재배 적합성과 높은 관련성을 보였습니다.

또한 여러 온도 관련 변수 사이에서 높은 상관관계가 확인되어, 단순한 선형 모델보다 비선형 관계를 학습할 수 있는 트리 기반 모델의 필요성을 확인했습니다.

2. 수익성과 재배 가능성은 별개의 문제

수익성 점수가 높은 작물이라고 해서 국내에서 재배하기 쉬운 것은 아니었습니다.

수익성 5점 작물의 평균 재배 적합률도 **7.5%**에 불과했습니다.

따라서 최종 추천 과정에서는

수익성
  +
재배 가능성
  +
지역 기후조건

을 함께 고려해야 합니다.

머신러닝 모델 비교

초기에는 다음 4개 모델을 비교했습니다.

Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
초기 모델 성능
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Random Forest	0.9985	0.9850	0.9924	0.9887	0.9999
Gradient Boosting	0.9980	0.9776	0.9924	0.9850	0.9998
Decision Tree	0.9974	0.9774	0.9848	0.9811	0.9916
Logistic Regression	0.9735	0.7174	1.0000	0.8354	0.9983

초기 비교에서는 Random Forest가 가장 높은 F1-score를 기록했습니다.

모델 고도화
Random Forest Hyperparameter Tuning

Random Forest를 대상으로 GridSearchCV를 적용했습니다.

GridSearchCV
    │
    ├── n_estimators
    │     └── 100 / 200 / 300
    │
    ├── max_depth
    │     └── None / 8 / 12
    │
    └── min_samples_leaf
          └── 1 / 3 / 5

최적 파라미터:

n_estimators = 100
max_depth = 12
min_samples_leaf = 1

튜닝 후 F1-score는 0.9886, ROC-AUC는 0.9999였습니다.

Hybrid Stacking

추가적인 모델 고도화를 위해 XGBoost와 LightGBM을 포함한 Hybrid Stacking 구조도 적용했습니다.

또한 Stratified 5-Fold Cross Validation을 사용하여 일반화 성능을 검증했습니다.

고도화 과정에서는 단순한 온도값보다

temp_diff_from_opt
frost_safety_margin

과 같은 파생변수가 중요한 변수로 나타났습니다.

Feature Importance

Random Forest 분석 결과 한계생육온도(frost_limit_temp)가 가장 중요한 변수로 나타났습니다.

주요 Feature

1. frost_limit_temp
2. opt_temp_min / opt_temp_max
3. min_temp
4. 기타 기후·생육조건

특히 온도 관련 변수가 전체 변수 중요도의 약 **78%**를 차지하여, 국내 환경에서는 온도 조건이 재배 가능성을 결정하는 핵심 요인임을 확인했습니다.

서비스 동작 구조

최종 서비스는 사용자가 다음 정보를 입력하는 방식으로 설계했습니다.

사용자 입력
│
├── 재배 지역
├── 재배 면적
└── 예상 투자금
        │
        ▼
지역 × 100개 작물 조합 생성
        │
        ▼
Random Forest
재배 적합성 예측
        │
        ▼
적합 작물 필터링
        │
        ▼
수익성 점수 기준 정렬
        │
        ▼
추천 작물 Best 7
        │
        ▼
기후 / 생육조건 / 경제성 정보 제공

Streamlit에서는 지역의 기후 프로필과 추천 작물의 생육조건을 비교하는 레이더 차트와 수익성·난이도 비교 그래프를 제공하도록 설계했습니다.

주요 기능
지역별 작물 추천

사용자가 재배 지역을 선택하면 해당 지역의 기후조건을 기준으로 100개 작물의 재배 적합성을 예측합니다.

수익성 기반 추천

재배 적합 판정을 받은 작물 가운데 수익성 점수를 기준으로 상위 작물을 선별합니다.

기후 프로필 시각화

선택한 지역의 기후조건과 작물의 생육조건을 시각적으로 비교합니다.

모델 정보 제공

모델의

Precision
Recall
F1-score
Feature Importance

등을 제공하여 추천 결과의 근거를 확인할 수 있도록 구성했습니다.

프로젝트 구조
climate-crop-recommendation/
│
├── data/
│   ├── korea10year.csv
│   ├── exotic_crops_100_environment_and_economy.csv
│   └── processed_ml_dataset.csv
│
├── notebook/
│   └── modeling.ipynb
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
│
└── images/
    ├── eda.png
    ├── model_comparison.png
    └── streamlit.png
Tech Stack

Language

Python

Data Analysis

Pandas
NumPy
Matplotlib
Seaborn

Machine Learning

Scikit-learn
Random Forest
Gradient Boosting
XGBoost
LightGBM

Model Evaluation

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Feature Importance
SHAP

Deployment / Visualization

Streamlit
GitHub
한계점

현재 모델은 과거 10년의 기후 데이터를 기반으로 현재의 재배 적합성을 예측합니다.

따라서 미래 기후변화에 따른 재배 가능성을 직접 예측하는 모델은 아닙니다.

또한 수익성 점수와 국내시장성 점수는 전문가 평가 기반의 서열척도이므로 실제 투자금 대비 순수익을 정확하게 계산하기에는 한계가 있습니다.

향후 발전 방향
1. 미래 기후 예측

LSTM / GRU 기반 시계열 모델을 활용하여 향후 3년의 지역별 기후를 예측하고, 미래 기후를 기준으로 작물 추천이 가능하도록 확장합니다.

2. 수익성 예측 고도화

실제 작물별

평당 수확량
시장가격
재배비용

데이터를 추가하여 사용자의 면적과 투자금을 반영한 실제 예상 순수익 예측 모델로 발전시킬 계획입니다.

3. 이미지 기반 분석

향후 재배지의 토양 및 환경 사진을 입력받아 토양 상태를 보조적으로 분석하는 기능을 추가할 수 있습니다.

프로젝트를 통해 확인한 점

본 프로젝트를 통해 "수익성이 높은 작물"과 "재배할 수 있는 작물"은 서로 다른 문제라는 점을 확인했습니다.

특히 국내 열대·아열대 작물의 재배 가능성은 수익성보다 동파 위험과 생육적온 충족 여부에 크게 영향을 받았습니다.

따라서 실제 추천 서비스에서는

재배 가능성 판단 → 경제성 평가 → 사용자 조건 반영

의 단계적인 접근이 필요합니다.

