import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="스마트 농가 맞춤형 작물 추천",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 데이터 로드 및 모델 학습 (캐싱)
# ==========================================
@st.cache_data
def load_and_prepare_model():
    df = pd.read_csv('./dataset/processed_ml_dataset.csv', encoding='utf-8-sig')
    
    rename_map = {
        '작물명': 'crop_name',
        '생육적온_최저(℃)': 'opt_temp_min',
        '생육적온_최고(℃)': 'opt_temp_max',
        '한계생육온도(℃)': 'frost_limit_temp',
        '적정습도(%)': 'opt_humidity',
        '토양pH_최저': 'soil_ph_min',
        '토양pH_최고': 'soil_ph_max',
        '수익성(1-5)': 'profit_score'
    }
    df = df.rename(columns=rename_map)
    
    if 'category' not in df.columns:
        df['category'] = np.where(df['opt_temp_max'] > 28, '과수', '일반작물')
        
    if 'cost_per_pyeong' not in df.columns:
        np.random.seed(42)
        df['cost_per_pyeong'] = np.random.choice([1.5, 2.0, 3.5, 5.0, 8.0], size=len(df))

    feature_cols = ['min_temp', 'max_temp', 'avg_temp', 'avg_rhm', 'annual_rn',
                    'opt_temp_min', 'opt_temp_max', 'frost_limit_temp', 'opt_humidity',
                    'soil_ph_min', 'soil_ph_max']
    
    X = df[feature_cols]
    y = df['suitability']
    
    model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    return df, model, feature_cols

try:
    df, model, feature_cols = load_and_prepare_model()
except Exception as e:
    st.error(f"데이터셋 및 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# ==========================================
# 3. 사이드바 - 농가 조건 입력
# ==========================================
with st.sidebar:
    st.header("📋 농가 조건 입력")
    
    regions = sorted(df['region'].dropna().unique().tolist()) if 'region' in df.columns else ['용인시 처인구 남사읍']
    selected_region = st.selectbox("1. 재배 지역 선택", options=regions)
    
    area_pyeong = st.number_input("2. 재배 면적 (평)", min_value=100, max_value=100000, value=1000, step=100)
    
    budget_max = st.number_input("3. 투자가능 / 예상 비용 (만원)", min_value=100, max_value=100000, value=3000, step=500)
    
    selected_category = st.radio("4. 희망 작물 분류", ["전체", "과수", "일반작물"], horizontal=True)
    
    st.divider()
    search_button = st.button("🔍 맞춤 작물 추천받기", type="primary", use_container_width=True)

# ==========================================
# 4. 메인 화면 출력
# ==========================================
st.title("🌾 AI 기반 맞춤형 작물 & 과수 추천 서비스")
st.caption("지역별 과거 기후 기록과 영농 예산을 종합 분석하여 최적의 작물과 리스크 정보를 제공합니다.")

st.divider()

if search_button:
    # 1. 기후 및 데이터 분석
    region_data = df[df['region'] == selected_region].iloc[0] if 'region' in df.columns else df.iloc[0]
    reg_min_temp = region_data['min_temp']
    reg_avg_temp = region_data['avg_temp']
    
    candidate_df = df.drop_duplicates(subset=['crop_name']).copy() if 'crop_name' in df.columns else df.copy()
    if selected_category != "전체":
        candidate_df = candidate_df[candidate_df['category'] == selected_category]
        
    candidate_df['estimated_total_cost'] = candidate_df['cost_per_pyeong'] * area_pyeong
    
    X_input = candidate_df[feature_cols]
    candidate_df['pred_proba'] = model.predict_proba(X_input)[:, 1]
    candidate_df['budget_fit'] = candidate_df['estimated_total_cost'] <= budget_max
    
    # 추천 점수 계산
    candidate_df['recommend_score'] = (
        candidate_df['pred_proba'] * 0.5 + 
        (candidate_df['profit_score'] / 5.0) * 0.3 + 
        (candidate_df['budget_fit'].astype(int)) * 0.2
    )
    
    top_recommendations = candidate_df.sort_values(by='recommend_score', ascending=False).head(3)
    
    # [상단 요약 메트릭]
    st.subheader(f"📍 {selected_region} 분석 브리핑")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("과거 최저기온", f"{reg_min_temp} ℃")
    m2.metric("연평균 기온", f"{reg_avg_temp} ℃")
    m3.metric("설정 면적", f"{area_pyeong:,} 평")
    m4.metric("입력 예산", f"{budget_max:,} 만원")
    
    st.write("")
    
    # [탭으로 정보 분리]
    tab1, tab2, tab3 = st.tabs(["🎯 TOP 3 추천 작물", "📊 비용 대 수익성 비교", "❄️ 한파 리스크 분석"])
    
    # --- TAB 1: 추천 작물 카드 ---
    with tab1:
        for idx, (_, row) in enumerate(top_recommendations.iterrows(), 1):
            crop_name = row.get('crop_name', f'작물 {idx}')
            profit = row['profit_score']
            total_cost = row['estimated_total_cost']
            cost_pyeong = row['cost_per_pyeong']
            proba = int(row['pred_proba'] * 100)
            
            # 리스크 조건 체크
            temp_margin = reg_min_temp - row['frost_limit_temp']
            is_temp_risk = temp_margin < 2.0
            is_budget_over = total_cost > budget_max
            
            # Streamlit 내장 컨테이너 적용
            with st.container(border=True):
                st.subheader(f"#{idx}. {crop_name} ({row.get('category', '작물')})")
                
                col_a, col_b, col_c = st.columns([1, 1, 1.2])
                
                with col_a:
                    st.write("**AI 기후 적합도**")
                    st.progress(proba / 100)
                    st.caption(f"적합 확률: **{proba}%**")
                    
                with col_b:
                    st.write("**비용 및 수익성**")
                    st.write(f"· 예상 수익성: {'★' * int(profit)}{'☆' * (5 - int(profit))} ({profit}/5)")
                    st.write(f"· 총 예상 비용: **{total_cost:,.0f} 만원**")
                    st.caption(f"(평당 {cost_pyeong:.1f} 만원)")
                    
                with col_c:
                    st.write("**💡 추천 이유 및 특징**")
                    st.caption(f"- 해당 지역의 최저기온({reg_min_temp}℃) 및 생육적온({row['opt_temp_min']}~{row['opt_temp_max']}℃)에 부합합니다.")
                    st.caption(f"- 적정 토양 pH: {row['soil_ph_min']} ~ {row['soil_ph_max']}")
                
                # 리스크 알림 메시지 (Streamlit 내장 warning/success/error 메시지 박스)
                if is_temp_risk or is_budget_over:
                    if is_temp_risk:
                        st.warning(f"⚠️ **한파/동해 주의**: 지역 최저기온({reg_min_temp}℃)이 한계온도({row['frost_limit_temp']}℃)와 가까우므로 보온 시설 보강을 권장합니다.", icon="⚠️")
                    if is_budget_over:
                        st.error(f"💸 **예산 초과 경고**: 예상 총비용({total_cost:,.0f}만원)이 설정한 예산({budget_max:,.0f}만원)을 {total_cost - budget_max:,.0f}만원 초과합니다.", icon="🚨")
                else:
                    st.success("✅ **안정적인 환경**: 기후 조건 및 예산 범위 내에 부합하는 안전한 선택입니다.", icon="✅")

    # --- TAB 2: 비용 vs 수익성 산점도 ---
    with tab2:
        st.subheader("비용 대비 수익성 분포")
        fig = px.scatter(
            top_recommendations,
            x='estimated_total_cost',
            y='profit_score',
            size='recommend_score',
            color='crop_name',
            text='crop_name',
            labels={'estimated_total_cost': '총 예상 비용 (만원)', 'profit_score': '수익성 점수 (1-5)'},
            template="plotly_white",
            height=450
        )
        fig.update_traces(textposition='top center')
        fig.add_vline(x=budget_max, line_dash="dash", line_color="red", annotation_text="내 예산 한도")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: 온도 비교 차트 ---
    with tab3:
        st.subheader("작물 한계 생육온도 vs 지역 최저기온")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['frost_limit_temp'],
            name='작물 한계 생육온도 (℃)',
            marker_color='#2b5c8f'
        ))
        fig_bar.add_hline(y=reg_min_temp, line_dash="dot", line_color="red", annotation_text=f"지역 최저기온 ({reg_min_temp}℃)")
        fig_bar.update_layout(template="plotly_white", height=400, yaxis_title="온도 (℃)")
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 조건을 선택한 후 **[🔍 맞춤 작물 추천받기]** 버튼을 눌러주세요.")