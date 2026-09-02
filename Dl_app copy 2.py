###DL 1차모델 버젼 steamlit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image

# XGBoost 머신러닝 모델 임포트 (model_training_xg.py)
from xgboost import XGBRegressor

# ==========================================
# 3. 2차 딥러닝 이미지 분석 모델
# ==========================================

import torch
import torch.nn as nn
import timm

from torchvision import transforms


# ==========================================
# 1. 페이지 기본 설정 및 CSS Customization
# ==========================================
st.set_page_config(
    page_title="기후 적응형 아열대작물·과수 추천 및 AI 분석 서비스",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #10B981;
        margin-bottom: 0px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 6px 0px;
        border-bottom: 2px solid #4B5563;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(156, 163, 175, 0.15);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 22px;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #4B5563;
        border: 1px solid rgba(156, 163, 175, 0.3);
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border: 1px solid #059669 !important;
        box-shadow: 0px -2px 8px rgba(16, 185, 129, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. XGBoost 모델 및 학명 파일 연동 로드 (캐싱)
# ==========================================
@st.cache_data
def load_xgboost_engine():
    # 1. 메인 기후 및 ML 데이터셋 로드
    df = pd.read_csv('./dataset/processed_ml_dataset.csv', encoding='utf-8-sig')
    
    # 2. 과수/작물 학명 정보 crop_master_100.csv 파일 로드 및 카테고리 강제 매핑
    try:
        master_df = pd.read_csv('./dataset/crop_master_100.csv', encoding='utf-8-sig')
        # master_df 컬럼: ID, crop_name_ko, crop_name_en, scientific_name, category, dataset_alias
        master_df = master_df.rename(columns={
            'crop_name_ko': 'crop_name',
            'category': 'master_category'
        })
        
        # 메인 데이터셋에 crop_name을 기준으로 마스터 파일의 정확한 과수/작물 구분 병합
        if 'crop_name' in df.columns and 'crop_name' in master_df.columns:
            df = pd.merge(df, master_df[['crop_name', 'master_category', 'crop_name_en', 'scientific_name']], on='crop_name', how='left')
            # 마스터 파일의 '과수' / '작물' 분류를 category 컬럼에 완전히 덮어씀
            df['category'] = df['master_category'].fillna(df.get('category', '작물'))
    except Exception as e:
        # 마스터 파일 로드 실패 시 기본 처리
        if 'category' not in df.columns:
            df['category'] = '작물'

    rename_map = {
        '작물명': 'crop_name',
        '생육적온_최저(℃)': 'opt_temp_min',
        '생육적온_최고(℃)': 'opt_temp_max',
        '한계생육온도(℃)': 'frost_limit_temp',
        '적정습도(%)': 'opt_humidity',
        '토양pH_최저': 'soil_ph_min',
        '토양pH_최고': 'soil_ph_max',
        '수익성(1-5)': 'profit_score',
        '수익성': 'profit_score',
        '평당비용(만원)': 'cost_per_pyeong',
        '평당비용': 'cost_per_pyeong',
        '평당경영비': 'cost_per_pyeong',
        '평당설비비': 'cost_per_pyeong',
    }
    df = df.rename(columns=rename_map)

    if 'cost_per_pyeong' not in df.columns:
        np.random.seed(42)
        df['cost_per_pyeong'] = np.random.uniform(2.0, 5.5, size=len(df)).round(1)
    else:
        df['cost_per_pyeong'] = pd.to_numeric(df['cost_per_pyeong'], errors='coerce').fillna(3.0)

    if 'opt_temp_avg' not in df.columns:
        df['opt_temp_avg'] = (df['opt_temp_min'] + df['opt_temp_max']) / 2.0
    if 'temp_diff_from_opt' not in df.columns:
        df['temp_diff_from_opt'] = np.abs(df['avg_temp'] - df['opt_temp_avg'])
    if 'frost_safety_margin' not in df.columns:
        df['frost_safety_margin'] = df['min_temp'] - df['frost_limit_temp']
    if 'profit_score' not in df.columns:
        df['profit_score'] = 3

    feature_cols = [
        'min_temp', 'max_temp', 'avg_temp', 'avg_rhm', 'annual_rn',
        'opt_temp_min', 'opt_temp_max', 'frost_limit_temp', 'opt_humidity',
        'soil_ph_min', 'soil_ph_max', 'temp_diff_from_opt', 'frost_safety_margin'
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df['suitability_score'] if 'suitability_score' in df.columns else df['suitability'] * 100
    
    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    
    return df, model, feature_cols

try:
    df, model, feature_cols = load_xgboost_engine()
except Exception as e:
    st.error(f"데이터셋/XGBoost 모델 로드 실패: {e}")
    st.stop()
# ==========================================
# 3. 2차 딥러닝 이미지 분석 모델
# ==========================================

import torch
import torch.nn as nn
import timm
from torchvision import transforms


# ------------------------------------------
# 2차 모델 기본 설정
# ------------------------------------------

DL_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DL_MODEL_PATH = "./model/best_plant_health_model.pth"

DL_CLASS_NAMES = [
    "정상",
    "질병",
    "질소 결핍",
    "인 결핍",
    "칼륨 결핍",
    "환경 스트레스"
]


# ------------------------------------------
# 2차 모델 구조
# 학습 코드와 동일하게 구성
# ------------------------------------------

class PlantHealthModel(nn.Module):

    def __init__(self):

        super().__init__()

        # EfficientNet-B3 Backbone
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        feature_dim = self.backbone.num_features

        # 학습 당시 classifier와 동일
        self.classifier = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                feature_dim,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                6
            )
        )


    def forward(self, x):

        features = self.backbone(x)

        output = self.classifier(features)

        return output


# ------------------------------------------
# 모델 로드
# ------------------------------------------

@st.cache_resource
def load_dl_model():

    model = PlantHealthModel()

    checkpoint = torch.load(
        DL_MODEL_PATH,
        map_location=DL_DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DL_DEVICE)

    model.eval()

    return model


# ------------------------------------------
# 모델 실행
# ------------------------------------------

try:

    dl_model = load_dl_model()

    dl_model_loaded = True

except Exception as e:

    dl_model = None
    dl_model_loaded = False

    st.error(
        f"2차 딥러닝 모델 로드 실패: {e}"
    )


# ------------------------------------------
# 이미지 전처리
# 학습 코드와 동일
# ------------------------------------------

dl_transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------
# 이미지 예측 함수
# ------------------------------------------

def predict_plant_image(image):

    # RGB 변환
    image = image.convert("RGB")

    # 전처리
    image_tensor = dl_transform(image)

    # Batch 차원 추가
    image_tensor = image_tensor.unsqueeze(0)

    # CPU / GPU 이동
    image_tensor = image_tensor.to(DL_DEVICE)

    # 예측
    with torch.no_grad():

        outputs = dl_model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    confidence_value = confidence.item()

    class_probabilities = (
        probabilities[0]
        .cpu()
        .numpy()
    )

    return (
        DL_CLASS_NAMES[predicted_index],
        confidence_value,
        class_probabilities
    )
# ==========================================
# 3. 사이드바 - 지역 검색 및 '과수' / '작물' 명확한 분류 선택
# ==========================================
with st.sidebar:
    st.header("📋 농가 조건 입력")
    
    all_raw_regions = sorted(list(df['region'].dropna().unique())) if 'region' in df.columns else ["수원시"]
    
    search_query = st.text_input("1. 지역 검색 (시·군·구)", value="수원", placeholder="예: 강릉, 강진, 수원, 제주...")
    clean_kw = search_query.strip().replace(" ", "")
    
    matched_regions = [r for r in all_raw_regions if clean_kw in r.replace(" ", "")]
    if not matched_regions:
        matched_regions = all_raw_regions

    selected_display_region = st.selectbox("📍 검색된 지역 선택", options=matched_regions, index=0)
    
    area_pyeong = st.number_input("2. 재배 면적 (평)", min_value=100, max_value=100000, value=500, step=50)
    user_budget = st.number_input("3. 투자 예산 (만원)", min_value=100, max_value=100000, value=2000, step=100)
    
    # 마스터 파일 기반으로 카테고리는 딱 '전체', '과수', '작물'로 깔끔하게 구성
    category_options = ["전체", "과수", "작물"]
    selected_category = st.radio("4. 희망 작물 분류 선택", options=category_options, horizontal=True)
    
    st.divider()
    search_button = st.button("🔍 AI 분석 추천 실행", type="primary", use_container_width=True)

# ==========================================
# 4. 메인 최상단 타이틀 및 2대 핵심 탭 구조
# ==========================================
st.markdown('<p class="main-header">🌾 스마트 농업 AI 통합 솔루션</p>', unsafe_allow_html=True)
st.caption("기후 데이터 기반 작물 추천 및 딥러닝 기반 이미지로 작물 건강 상태 분석 서비스입니다.")

st.write("")

tab1, tab2 = st.tabs([
    "🌿 기후 적응형 작물·과수 추천 대시보드", 
    "📷 AI 작물 건강 상태 진단"
])

# ==========================================
# [TAB 1] 기후 적응형 작물/과수 추천 (마스터 분류 반영)
# ==========================================
with tab1:
    st.write("")
    region_data = df[df['region'] == selected_display_region].iloc[0] if 'region' in df.columns else df.iloc[0]
    reg_min_temp = region_data['min_temp']
    reg_avg_temp = round(float(region_data['avg_temp']), 1)
    
    candidate_df = df.drop_duplicates(subset=['crop_name']).copy()
    
    # 마스터 파일에서 매핑된 정확한 category 컬럼을 기준으로 필터링
    if selected_category != "전체" and 'category' in candidate_df.columns:
        candidate_df = candidate_df[candidate_df['category'].astype(str).str.strip() == selected_category]
        
    candidate_df['min_temp'] = reg_min_temp
    candidate_df['max_temp'] = region_data['max_temp']
    candidate_df['avg_temp'] = region_data['avg_temp']
    candidate_df['avg_rhm'] = region_data['avg_rhm']
    candidate_df['annual_rn'] = region_data['annual_rn']
    
    candidate_df['opt_temp_avg'] = (candidate_df['opt_temp_min'] + candidate_df['opt_temp_max']) / 2.0
    candidate_df['temp_diff_from_opt'] = np.abs(candidate_df['avg_temp'] - candidate_df['opt_temp_avg'])
    candidate_df['frost_safety_margin'] = candidate_df['min_temp'] - candidate_df['frost_limit_temp']
    
    candidate_df['ai_suitability_score'] = model.predict(candidate_df[feature_cols])
    candidate_df['ai_suitability_score'] = candidate_df['ai_suitability_score'].clip(0, 100).round(1)
    
    candidate_df['estimated_total_cost'] = candidate_df['cost_per_pyeong'] * area_pyeong
    
    top_recommendations = candidate_df.sort_values(by='ai_suitability_score', ascending=False).head(7).reset_index(drop=True)
    
    # 지역 브리핑 메트릭
    st.subheader(f"📍 [{selected_display_region}] 지역 기후 진단 브리핑 (선택 분류: {selected_category})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("선택 지역 최저기온", f"{reg_min_temp} ℃")
    m2.metric("선택 지역 연평균기온", f"{reg_avg_temp} ℃")
    m3.metric("설정 재배 면적", f"{area_pyeong:,} 평")
    m4.metric("입력 투자 예산", f"{user_budget:,} 만원")
    
    st.write("")
    st.divider()

    # 1. TOP 7 추천 작물/과수 카드 리스트
    st.subheader(f"🎯 [{selected_category}] 분석 TOP 7 베스트 추천 리스트")
    st.markdown(f"마스터 데이터베이스 기준에 따라 완벽하게 분류된 **[{selected_category}]** 품목 중 최적의 순위입니다.")
    
    if len(top_recommendations) == 0:
        st.warning(f"⚠️ 선택하신 지역({selected_display_region})의 기후 조건에 부합하는 [{selected_category}] 데이터가 없습니다. 다른 분류나 지역을 선택해 주세요.")
    else:
        for idx, row in top_recommendations.iterrows():
            crop_name = row['crop_name']
            en_name = row.get('crop_name_en', '')
            sci_name = row.get('scientific_name', '')
            profit = int(row['profit_score'])
            ai_score = row['ai_suitability_score']
            margin = row['frost_safety_margin']
            cost_py = row['cost_per_pyeong']
            total_cost = row['estimated_total_cost']
            
            is_temp_risk = margin < 2.0
            is_budget_over = total_cost > user_budget
            
            with st.container(border=True):
                st.markdown(f"### **#{idx+1}. {crop_name}** `({en_name} / {sci_name})`")
                
                col_a, col_b, col_c = st.columns([1.2, 1, 1.2])
                
                with col_a:
                    st.markdown("**기후 정밀 적합도**")
                    st.progress(min(1.0, max(0.0, ai_score / 100.0)))
                    st.markdown(f"AI 적합 점수: <span style='font-size:1.25rem; font-weight:bold; color:#10B981;'>{ai_score:.1f}점</span>", unsafe_allow_html=True)
                    
                with col_b:
                    st.markdown("**💰 경영 및 비용 지표**")
                    st.markdown(f"• 평당 비용: **약 {cost_py:,.1f} 만원**")
                    st.markdown(f"• 총 추정 비용: <span style='font-size:1.1rem; color:#3B82F6; font-weight:bold;'>약 {total_cost:,.0f} 만원</span> ({area_pyeong:,}평)", unsafe_allow_html=True)
                    st.markdown(f"• 예상 수익성: <span style='color:#F59E0B;'>**{'★' * profit}{'☆' * (5 - profit)}**</span> ({profit}/5)", unsafe_allow_html=True)
                    
                with col_c:
                    st.markdown("**💡 AI 예측 주요 피처 요인**")
                    st.markdown(f"- 최저기온 여유 마진: **{margin:+.1f}℃**")
                    st.markdown(f"- 생육 적온 중심과의 격차: **{row['temp_diff_from_opt']:.1f}℃**")
                
                st.write("")
                if is_budget_over:
                    st.warning(f"💡 **예산 초과 주의**: 추정 비용({total_cost:,.0f}만원)이 설정하신 예산({user_budget:,.0f}만원)을 초과합니다.", icon="💡")
                elif is_temp_risk:
                    st.warning(f"⚠️ **한파 주의**: 지역 최저기온({reg_min_temp}℃) 대비 작물 한계온도({row['frost_limit_temp']}℃) 여유 마진({margin:+.1f}℃)이 적습니다.", icon="⚠️")
                else:
                    st.success("✅ **안정적 생육 및 예산 조건 충족**: 기후 조건 및 예산 범위 적합.", icon="✅")

    st.write("")
    st.divider()

    # 2. 예산 비교 시각화 차트
    st.subheader(f"📊 [예산 비교] {area_pyeong:,}평 기준 품목별 실제 예상 비용 vs 입력 예산 ({user_budget:,}만원)")
    
    if len(top_recommendations) > 0:
        bar_colors = ['#EF4444' if cost > user_budget else '#3B82F6' for cost in top_recommendations['estimated_total_cost']]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['estimated_total_cost'],
            name='실제 예상 비용 (만원)',
            marker_color=bar_colors,
            text=top_recommendations['estimated_total_cost'].apply(lambda x: f"{x:,.0f} 만원"),
            textposition='outside'
        ))
        
        fig2.add_hline(
            y=user_budget, 
            line_dash="dash", 
            line_color="#10B981", 
            line_width=3,
            annotation_text=f"내 설정 예산 ({user_budget:,} 만원)",
            annotation_position="top right",
            annotation_font=dict(size=14, color="#10B981")
        )
        
        max_val = max(top_recommendations['estimated_total_cost'].max(), user_budget) * 1.25
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=480, bargap=0.45,
            xaxis=dict(title=dict(text="품목명", font=dict(size=14)), tickfont=dict(size=13), showgrid=False),
            yaxis=dict(title=dict(text="비용 (만원)", font=dict(size=13)), gridcolor='rgba(156,163,175,0.2)', range=[0, max_val]),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.write("")
    st.divider()

    # 3. 한파 리스크 진단 시각화 차트
    st.subheader("❄️ [한파 리스크] ")
    
    if len(top_recommendations) > 0:
        colors = ['#10B981' if m >= 3.0 else ('#F59E0B' if m >= 0 else '#EF4444') for m in top_recommendations['frost_safety_margin']]
        
        fig3 = go.Figure(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['frost_safety_margin'],
            marker_color=colors,
            text=top_recommendations['frost_safety_margin'].apply(lambda x: f"{x:+.1f} ℃"),
            textposition='outside'
        ))
        
        fig3.add_hline(y=0, line_dash="solid", line_color="#6B7280", annotation_text="동해 위험 기준선 (0℃)")
        
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, bargap=0.45,
            xaxis=dict(tickfont=dict(size=13), showgrid=False),
            yaxis=dict(title=dict(text="여유 마진 (℃)", font=dict(size=13)), gridcolor='rgba(156,163,175,0.2)'),
            showlegend=False, margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# [TAB 2] 딥러닝 이미지 진단
# ==========================================
# ==========================================
# [TAB 2] 2차 딥러닝 식물 건강 상태 분석
# ==========================================

with tab2:

    st.write("")

    st.subheader(
        "📷 딥러닝 기반 식물 건강 상태 분석"
    )

    st.info(
        "식물 잎 이미지를 분석하여 "
        "정상, 질병, 영양 결핍 및 환경 스트레스 상태를 예측합니다."
    )

    st.write("")

    # ------------------------------------------
    # 이미지 업로드
    # ------------------------------------------

    col_upload, col_preview = st.columns(
        [1, 1.2],
        gap="large"
    )


    with col_upload:

        st.markdown(
            "#### 1. 진단 대상 이미지 업로드"
        )

        uploaded_file = st.file_uploader(
            "농작물 잎 또는 과수 잎 사진을 업로드하세요.",
            type=["jpg", "jpeg", "png"]
        )

        st.write("")

        run_dl_button = st.button(
            "🔬 딥러닝 AI 진단 실행",
            type="primary",
            use_container_width=True
        )


    # ------------------------------------------
    # 이미지 미리보기
    # ------------------------------------------

    with col_preview:

        st.markdown(
            "#### 2. 이미지 미리보기"
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="업로드된 진단 이미지",
                use_container_width=True
            )

        else:

            st.markdown(
                """
                <div style="
                    border: 2px dashed rgba(156,163,175,0.4);
                    border-radius: 10px;
                    padding: 40px;
                    text-align: center;
                    color: #9CA3AF;
                ">

                    <p style="
                        font-size: 1.2rem;
                        font-weight: 600;
                    ">
                        이미지가 업로드되면
                        여기에 표시됩니다.
                    </p>

                    <p style="
                        font-size: 0.95rem;
                    ">
                        좌측에서 식물 이미지를 선택해주세요.
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ------------------------------------------
    # AI 진단 실행
    # ------------------------------------------

    if run_dl_button:

        if uploaded_file is None:

            st.warning(
                "먼저 진단할 식물 이미지를 업로드해주세요."
            )

        elif not dl_model_loaded:

            st.error(
                "2차 딥러닝 모델을 불러오지 못했습니다."
            )

            st.info(
                "model/best_plant_health_model.pth "
                "파일이 있는지 확인해주세요."
            )

        else:

            # 이미지 분석
            result_class, confidence, probabilities = (
                predict_plant_image(image)
            )

            st.divider()

            st.markdown(
                "### 🔬 딥러닝 AI 분석 결과"
            )


            # ------------------------------------------
            # 주요 결과
            # ------------------------------------------

            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.markdown(
                    "#### 판정 결과"
                )


                if result_class == "정상":

                    st.success(
                        f"🌿 **{result_class}**"
                    )

                elif result_class == "질병":

                    st.error(
                        f"⚠️ **{result_class} 의심**"
                    )

                elif "결핍" in result_class:

                    st.warning(
                        f"🍂 **{result_class} 의심**"
                    )

                elif result_class == "환경 스트레스":

                    st.warning(
                        f"🌡️ **{result_class} 의심**"
                    )

                else:

                    st.info(
                        f"❓ **{result_class}**"
                    )


            with result_col2:

                st.markdown(
                    "#### AI 예측 신뢰도"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.1f}%"
                )

                st.progress(
                    float(confidence)
                )


            st.write("")


            # ------------------------------------------
            # 클래스별 확률
            # ------------------------------------------

            st.markdown(
                "#### 📊 상태별 AI 예측 확률"
            )

            probability_df = pd.DataFrame({

                "식물 상태": DL_CLASS_NAMES,

                "예측 확률 (%)":
                    probabilities * 100

            })


            probability_df[
                "예측 확률 (%)"
            ] = probability_df[
                "예측 확률 (%)"
            ].round(1)


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


            st.write("")


            # ------------------------------------------
            # 상태별 설명
            # ------------------------------------------

            st.markdown(
                "#### 💡 AI 분석 해석"
            )


            if result_class == "정상":

                st.success(
                    "현재 이미지에서는 정상적인 식물 상태가 "
                    "가장 높게 예측되었습니다."
                )

            elif result_class == "질병":

                st.warning(
                    "잎의 병해 관련 특징이 관찰될 가능성이 있습니다. "
                    "추가적인 이미지 촬영을 통한 확인이 필요합니다."
                )

            elif result_class == "질소 결핍":

                st.warning(
                    "질소 결핍과 관련된 잎의 특징이 "
                    "나타날 가능성이 있습니다."
                )

            elif result_class == "인 결핍":

                st.warning(
                    "인 결핍과 관련된 잎의 특징이 "
                    "나타날 가능성이 있습니다."
                )

            elif result_class == "칼륨 결핍":

                st.warning(
                    "칼륨 결핍과 관련된 잎의 특징이 "
                    "나타날 가능성이 있습니다."
                )

            elif result_class == "환경 스트레스":

                st.warning(
                    "수분 부족, 고온·저온 등 환경적 스트레스와 "
                    "관련된 특징이 나타날 가능성이 있습니다."
                )


            st.write("")

            st.caption(
                "※ AI 분석 결과는 이미지에서 학습된 특징을 기반으로 한 "
                "예측값이며, 실제 농작물 상태와 차이가 있을 수 있습니다."
            )