import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Scikit-Learn 내장 그래디언트 부스팅 모델
from sklearn.ensemble import HistGradientBoostingRegressor

# ==========================================
# 1. 페이지 기본 설정 및 CSS Customization
# ==========================================
st.set_page_config(
    page_title="AI 스마트 농가 맞춤형 작물 추천 시스템",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 6px 0px;
        border-bottom: 2px solid #4B5563;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(156, 163, 175, 0.15);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 18px;
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
# 2. 고도화 모델 및 데이터 로드 (캐싱)
# ==========================================
@st.cache_data
def load_advanced_engine():
    df = pd.read_csv('./dataset/processed_ml_dataset.csv', encoding='utf-8-sig')
    
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
        '카테고리': 'category'
    }
    df = df.rename(columns=rename_map)

    # 평당 비용 컬럼이 없거나 결측치일 때 처리
    if 'cost_per_pyeong' not in df.columns:
        # 데이터에 따라 조금씩 다른 난수 또는 기본값 분배로 차등화
        np.random.seed(42)
        df['cost_per_pyeong'] = np.random.uniform(2.0, 5.5, size=len(df)).round(1)
    else:
        df['cost_per_pyeong'] = pd.to_numeric(df['cost_per_pyeong'], errors='coerce').fillna(3.0)

    # 필수 파생 피처 생성
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
    
    model = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    
    return df, model, feature_cols

try:
    df, model, feature_cols = load_advanced_engine()
except Exception as e:
    st.error(f"데이터셋/모델 로드 실패: {e}")
    st.stop()

REGION_MAP = {
    "광주": ["경기도 광주시", "광주광역시"],
    "고성": ["강원특별자치도 고성군", "경상남도 고성군"],
    "오포": ["경기도 광주시"], "경안": ["경기도 광주시"], "곤지암": ["경기도 광주시"], "초월": ["경기도 광주시"],
    "치평": ["광주광역시"], "상무": ["광주광역시"], "첨단": ["광주광역시"], "수완": ["광주광역시"],
    "춘천": ["강원특별자치도 춘천시"], "금산": ["충청남도 금산군"], "강진": ["전라남도 강진군"], "수원": ["경기도 수원시"]
}

# ==========================================
# 3. 사이드바 - 농가 조건 입력 (복원 완료)
# ==========================================
with st.sidebar:
    st.header("📋 농가 조건 입력")
    raw_regions = sorted(list(df['region'].dropna().unique())) if 'region' in df.columns else ["수원"]
    
    search_query = st.text_input("1. 재배 지역/읍·면·동 검색", value="수원")
    clean_kw = search_query.strip().replace(" ", "")
    matched_full_names = []
    
    for key, full_list in REGION_MAP.items():
        if key in clean_kw or clean_kw in key:
            matched_full_names.extend(full_list)
            
    if not matched_full_names:
        for r in raw_regions:
            if clean_kw in r.replace(" ", ""):
                matched_full_names.append(r)
                
    matched_full_names = list(dict.fromkeys(matched_full_names))
    if not matched_full_names:
        matched_full_names = raw_regions

    selected_display_region = st.selectbox("📍 상세 지역 선택", options=matched_full_names, index=0)
    
    dataset_region_target = selected_display_region
    for r in raw_regions:
        if r in selected_display_region or selected_display_region in r:
            dataset_region_target = r
            break

    area_pyeong = st.number_input("2. 재배 면적 (평)", min_value=100, max_value=100000, value=500, step=50)
    
    # 💰 [복원] 투자 예상 비용 입력
    user_budget = st.number_input("3. 투자 예산 (만원)", min_value=100, max_value=100000, value=2000, step=100)
    
    categories = ["전체"]
    if 'category' in df.columns:
        categories += list(df['category'].dropna().unique())
    selected_category = st.radio("4. 희망 작물 분류", categories, horizontal=True)
    
    st.divider()
    search_button = st.button("🔍 AI 고도화 추천 실행", type="primary", use_container_width=True)

# ==========================================
# 4. 메인 분석 및 시각화 영역
# ==========================================
st.title("🌾 AI 기반 맞춤형 작물 & 과수 추천 서비스")
st.caption("고도화 머신러닝 엔진이 기후 상호작용 지표와 투자 예산을 종합 분석하여 최적 작물을 추천합니다.")

st.divider()

if search_button or selected_display_region:
    region_data = df[df['region'] == dataset_region_target].iloc[0] if 'region' in df.columns else df.iloc[0]
    reg_min_temp = region_data['min_temp']
    reg_avg_temp = round(float(region_data['avg_temp']), 1)
    
    candidate_df = df.drop_duplicates(subset=['crop_name']).copy()
    if selected_category != "전체" and 'category' in candidate_df.columns:
        candidate_df = candidate_df[candidate_df['category'] == selected_category]
        
    candidate_df['min_temp'] = reg_min_temp
    candidate_df['max_temp'] = region_data['max_temp']
    candidate_df['avg_temp'] = region_data['avg_temp']
    candidate_df['avg_rhm'] = region_data['avg_rhm']
    candidate_df['annual_rn'] = region_data['annual_rn']
    
    candidate_df['opt_temp_avg'] = (candidate_df['opt_temp_min'] + candidate_df['opt_temp_max']) / 2.0
    candidate_df['temp_diff_from_opt'] = np.abs(candidate_df['avg_temp'] - candidate_df['opt_temp_avg'])
    candidate_df['frost_safety_margin'] = candidate_df['min_temp'] - candidate_df['frost_limit_temp']
    
    # AI 기후 적합도 예측
    candidate_df['ai_suitability_score'] = model.predict(candidate_df[feature_cols])
    candidate_df['ai_suitability_score'] = candidate_df['ai_suitability_score'].clip(0, 100).round(1)
    
    # 예상 총 비용 계산 (평당 비용 * 면적)
    candidate_df['estimated_total_cost'] = candidate_df['cost_per_pyeong'] * area_pyeong
    
    top_recommendations = candidate_df.sort_values(by='ai_suitability_score', ascending=False).head(7).reset_index(drop=True)
    
    st.subheader(f"📍 {selected_display_region} 분석 브리핑")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("과거 최저기온", f"{reg_min_temp} ℃")
    m2.metric("연평균 기온", f"{reg_avg_temp} ℃")
    m3.metric("설정 면적", f"{area_pyeong:,} 평")
    m4.metric("입력 예산", f"{user_budget:,} 만원")
    
    st.write("")
    
    tab1, tab2, tab3 = st.tabs(["🎯 TOP 7 추천 작물", "💰 예산 vs 실제 예상 비용 비교", "❄️ 한파 리스크 진단"])
    
    # --- TAB 1: 추천 작물 리스트 ---
    with tab1:
        st.write("")
        for idx, row in top_recommendations.iterrows():
            crop_name = row['crop_name']
            profit = int(row['profit_score'])
            ai_score = row['ai_suitability_score']
            margin = row['frost_safety_margin']
            cost_py = row['cost_per_pyeong']
            total_cost = row['estimated_total_cost']
            
            is_temp_risk = margin < 2.0
            is_budget_over = total_cost > user_budget
            
            with st.container(border=True):
                st.markdown(f"### **#{idx+1}. {crop_name}** `({row.get('category', '작물')})`")
                
                col_a, col_b, col_c = st.columns([1.2, 1, 1.2])
                
                with col_a:
                    st.markdown("**ML 엔진 기후 정밀 적합도**")
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

    # --- TAB 2: 예산 vs 실제 예상 비용 비교 차트 (개선) ---
    with tab2:
        st.subheader(f"📊 {area_pyeong:,}평 기준 작물별 실제 예상 비용 vs 입력 예산 ({user_budget:,}만원)")
        
        # 예산 초과 여부에 따라 색상 지정 (초과 시 주황색, 충족 시 푸른색)
        bar_colors = ['#EF4444' if cost > user_budget else '#3B82F6' for cost in top_recommendations['estimated_total_cost']]
        
        fig2 = go.Figure()
        
        # 1. 작물별 실제 예상 비용 막대 그래프
        fig2.add_trace(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['estimated_total_cost'],
            name='실제 예상 비용 (만원)',
            marker_color=bar_colors,
            text=top_recommendations['estimated_total_cost'].apply(lambda x: f"{x:,.0f} 만원"),
            textposition='outside'
        ))
        
        # 2. 농가 보유 예산 점선 가이드라인
        fig2.add_hline(
            y=user_budget, 
            line_dash="dash", 
            line_color="#10B981", 
            line_width=3,
            annotation_text=f"내 설정 예산 ({user_budget:,} 만원)",
            annotation_position="top right",
            annotation_font=dict(size=14, color="#10B981")
        )
        
        # Y축 범위를 예산과 예상비용 중 큰 값 기준으로 여유있게 설정
        max_val = max(top_recommendations['estimated_total_cost'].max(), user_budget) * 1.25
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=520, bargap=0.45,
            xaxis=dict(title=dict(text="추천 작물명", font=dict(size=14)), tickfont=dict(size=13), showgrid=False),
            yaxis=dict(title=dict(text="비용 (만원)", font=dict(size=13)), gridcolor='rgba(156,163,175,0.2)', range=[0, max_val]),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 3: 한파 안전 마진 차트 ---
    with tab3:
        st.subheader("❄️ 작물별 한파 안전 여유 온도 (Safety Margin)")
        
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
            height=500, bargap=0.45,
            xaxis=dict(tickfont=dict(size=13), showgrid=False),
            yaxis=dict(title=dict(text="여유 마진 (℃)", font=dict(size=13)), gridcolor='rgba(156,163,175,0.2)'),
            showlegend=False, margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)