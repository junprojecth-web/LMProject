
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 페이지 기본 설정 및 고대비 Custom CSS
# ==========================================
st.set_page_config(
    page_title="스마트 농가 맞춤형 작물 추천",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 라이트/다크 모드 공용 탭 및 글자 시안성 강화 CSS
st.markdown("""
<style>
    /* 탭 전체 Container 가시성 강화 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 8px 0px;
        border-bottom: 2px solid #4B5563;
    }

    /* 기본 탭 버튼 스타일 */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(156, 163, 175, 0.15);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #4B5563;
        border: 1px solid rgba(156, 163, 175, 0.3);
        border-bottom: none;
    }

    /* 선택된 활성 탭 버튼 스타일 */
    .stTabs [aria-selected="true"] {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
        border: 1px solid #059669 !important;
        border-bottom: none !important;
        box-shadow: 0px -2px 10px rgba(16, 185, 129, 0.3);
    }
    
    /* 카드 내 강조 텍스트 가시성 보정 */
    .stCardText {
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 데이터 로드 및 모델 학습
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
    
    if 'crop_name' not in df.columns:
        df['crop_name'] = df['ID'].astype(str) + "번 작물"
        
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
    st.error(f"데이터셋 로드 실패: {e}")
    st.stop()

REGION_MAP = {
    "광주": ["경기도 광주시", "광주광역시"],
    "고성": ["강원특별자치도 고성군", "경상남도 고성군"],
    "오포": ["경기도 광주시"], "경안": ["경기도 광주시"], "곤지암": ["경기도 광주시"], "초월": ["경기도 광주시"],
    "치평": ["광주광역시"], "상무": ["광주광역시"], "첨단": ["광주광역시"], "수완": ["광주광역시"],
    "춘천": ["강원특별자치도 춘천시"], "금산": ["충청남도 금산군"], "강진": ["전라남도 강진군"]
}

def calculate_detailed_suitability(row, reg_data):
    opt_temp_avg = (row['opt_temp_min'] + row['opt_temp_max']) / 2.0
    temp_diff = abs(reg_data['avg_temp'] - opt_temp_avg)
    temp_score = np.exp(-0.05 * (temp_diff ** 2)) * 100
    
    frost_margin = reg_data['min_temp'] - row['frost_limit_temp']
    frost_score = 100 if frost_margin >= 5 else max(0, (frost_margin + 5) * 10)
    
    final_score = (temp_score * 0.6) + (frost_score * 0.4)
    return round(float(final_score), 1)

# ==========================================
# 3. 사이드바 입력
# ==========================================
with st.sidebar:
    st.header("📋 농가 조건 입력")
    raw_regions = sorted(list(df['region'].dropna().unique())) if 'region' in df.columns else ["춘천"]
    
    search_query = st.text_input("1. 재배 지역/읍·면·동 검색", value="춘천")
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

    selected_display_region = st.selectbox("📍 상세 선택", options=matched_full_names, index=0)
    
    dataset_region_target = selected_display_region
    for r in raw_regions:
        if r in selected_display_region or selected_display_region in r:
            dataset_region_target = r
            break

    area_pyeong = st.number_input("2. 재배 면적 (평)", min_value=100, max_value=100000, value=1000, step=100)
    budget_max = st.number_input("3. 투자가능 / 예상 비용 (만원)", min_value=100, max_value=100000, value=3000, step=500)
    selected_category = st.radio("4. 희망 작물 분류", ["전체", "과수", "일반작물"], horizontal=True)
    
    st.divider()
    search_button = st.button("🔍 맞춤 작물 추천받기", type="primary", use_container_width=True)

# ==========================================
# 4. 메인 화면 출력
# ==========================================
st.title("🌾 AI 기반 맞춤형 작물 & 과수 추천 서비스")
st.caption("지역별 기후 기록과 영농 예산을 종합 분석하여 최적의 작물과 리스크 정보를 제공합니다.")

st.divider()

if search_button or selected_display_region:
    region_data = df[df['region'] == dataset_region_target].iloc[0] if 'region' in df.columns else df.iloc[0]
    reg_min_temp = region_data['min_temp']
    reg_avg_temp = round(float(region_data['avg_temp']), 1)
    
    candidate_df = df.drop_duplicates(subset=['crop_name']).copy()
    if selected_category != "전체":
        candidate_df = candidate_df[candidate_df['category'] == selected_category]
        
    candidate_df['estimated_total_cost'] = candidate_df['cost_per_pyeong'] * area_pyeong
    candidate_df['detailed_suitability'] = candidate_df.apply(
        lambda r: calculate_detailed_suitability(r, region_data), axis=1
    )
    candidate_df['budget_fit'] = candidate_df['estimated_total_cost'] <= budget_max
    
    candidate_df['recommend_score'] = (
        (candidate_df['detailed_suitability'] / 100.0) * 0.5 + 
        (candidate_df['profit_score'] / 5.0) * 0.3 + 
        (candidate_df['budget_fit'].astype(int)) * 0.2
    )
    
    top_recommendations = candidate_df.sort_values(by='recommend_score', ascending=False).head(7).reset_index(drop=True)
    
    st.subheader(f"📍 {selected_display_region} 분석 브리핑")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("과거 최저기온", f"{reg_min_temp} ℃")
    m2.metric("연평균 기온", f"{reg_avg_temp} ℃")
    m3.metric("설정 면적", f"{area_pyeong:,} 평")
    m4.metric("입력 예산", f"{budget_max:,} 만원")
    
    st.write("")
    
    # [시안성이 대폭 확장된 탭 구성]
    tab1, tab2, tab3 = st.tabs(["🎯 TOP 7 추천 작물", "📊 비용 vs 수익성 비교", "❄️ 한파 리스크 진단"])
    
    # --- TAB 1: 추천 작물 카드 (글자 가시성 강화) ---
    with tab1:
        st.write("")
        for idx, row in top_recommendations.iterrows():
            crop_name = row['crop_name']
            profit = int(row['profit_score'])
            total_cost = row['estimated_total_cost']
            cost_pyeong = row['cost_per_pyeong']
            suitability_pct = row['detailed_suitability']
            
            temp_margin = reg_min_temp - row['frost_limit_temp']
            is_temp_risk = temp_margin < 2.0
            is_budget_over = total_cost > budget_max
            
            with st.container(border=True):
                st.markdown(f"### **#{idx+1}. {crop_name}** `({row.get('category', '작물')})`")
                
                col_a, col_b, col_c = st.columns([1, 1, 1.2])
                
                with col_a:
                    st.markdown("**AI 세부 기후 적합도**")
                    st.progress(min(1.0, max(0.0, suitability_pct / 100.0)))
                    st.markdown(f"적합도: <span style='font-size:1.2rem; font-weight:bold; color:#10B981;'>{suitability_pct:.1f}%</span>", unsafe_allow_html=True)
                    
                with col_b:
                    st.markdown("**비용 및 수익성**")
                    st.markdown(f"• 예상 수익성: **{'★' * profit}{'☆' * (5 - profit)}** ({profit}/5)")
                    st.markdown(f"• 총 예상 비용: <span style='font-size:1.1rem; font-weight:bold;'>{total_cost:,.0f} 만원</span>", unsafe_allow_html=True)
                    st.caption(f"(평당 {cost_pyeong:.1f} 만원)")
                    
                with col_c:
                    st.markdown("**💡 추천 이유 및 환경 특징**")
                    st.markdown(f"- 지역 최저기온(**{reg_min_temp}℃**) 및 생육적온(**{row['opt_temp_min']}~{row['opt_temp_max']}℃**) 매칭")
                    st.markdown(f"- 적정 토양 pH 조건: **{row['soil_ph_min']} ~ {row['soil_ph_max']}**")
                
                st.write("")
                if is_temp_risk or is_budget_over:
                    if is_temp_risk:
                        st.warning(f"⚠️ **한파 주의**: 지역 최저기온({reg_min_temp}℃) 대비 작물 한계온도({row['frost_limit_temp']}℃) 여유 부족.", icon="⚠️")
                    if is_budget_over:
                        st.error(f"💸 **예산 초과**: 총 비용({total_cost:,.0f}만원)이 설정 예산을 {total_cost - budget_max:,.0f}만원 초과합니다.", icon="🚨")
                else:
                    st.success("✅ **안정적인 환경**: 기후 적합도 및 예산 조건에 모두 부합합니다.", icon="✅")

    # --- TAB 2: 화이트/블랙 공용 다성능 차트 ---
    with tab2:
        st.subheader("📊 TOP 7 작물별 총 예상 비용 및 수익성 비교")
        st.caption("파란색 막대는 총 예상 비용(만원), 초록색 라인은 수익성 점수(1~5점)입니다.")
        
        fig2 = go.Figure()
        
        # 1. 예상 비용 막대 차트
        fig2.add_trace(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['estimated_total_cost'],
            name='총 예상 비용 (만원)',
            marker_color='#3B82F6',
            text=top_recommendations['estimated_total_cost'].apply(lambda x: f"{x:,.0f}만원"),
            textposition='outside',
            textfont=dict(size=12, color='#2563EB'),
            yaxis='y'
        ))
        
        # 2. 수익성 점수 꺾은선 차트
        fig2.add_trace(go.Scatter(
            x=top_recommendations['crop_name'],
            y=top_recommendations['profit_score'],
            name='수익성 점수 (1-5)',
            marker=dict(size=10, color='#059669', symbol='diamond'),
            line=dict(width=3, color='#059669'),
            mode='lines+markers+text',
            text=top_recommendations['profit_score'].apply(lambda x: f"{x}점"),
            textposition='top center',
            textfont=dict(size=12, color='#059669'),
            yaxis='y2'
        ))
        
        # 예산 한도선
        fig2.add_hline(
            y=budget_max, 
            line_dash="dash", 
            line_color="#DC2626", 
            line_width=2,
            annotation_text=f"내 예산 한도 ({budget_max:,}만원)",
            annotation_position="top right",
            annotation_font=dict(color="#DC2626", size=12)
        )
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=580,
            bargap=0.45,
            xaxis=dict(
                title=dict(text="추천 작물명", font=dict(size=14, color="#374151")),
                tickfont=dict(size=13, color="#374151"),
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text="총 예상 비용 (만원)", font=dict(size=13, color="#2563EB")),
                tickfont=dict(size=12, color="#374151"),
                gridcolor='rgba(156, 163, 175, 0.2)',
                zeroline=False
            ),
            yaxis2=dict(
                title=dict(text="수익성 점수 (1-5)", font=dict(size=13, color="#059669")),
                tickfont=dict(size=12, color="#374151"),
                overlaying='y',
                side='right',
                range=[0, 6],
                showgrid=False,
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12)
            ),
            margin=dict(l=50, r=50, t=60, b=60)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 3: 화이트/블랙 공용 한파 안전 마진 차트 ---
    with tab3:
        st.subheader("❄️ 작물별 한파 안전 여유 온도 (Safety Margin)")
        st.caption("안전 여유 온도 = 지역 최저기온 - 작물 한계온도 (0℃ 이상일수록 동해에 안전합니다.)")
        
        top_recommendations['safety_margin'] = top_recommendations['frost_limit_temp'].apply(lambda x: reg_min_temp - x)
        colors = ['#10B981' if m >= 3.0 else ('#F59E0B' if m >= 0 else '#EF4444') for m in top_recommendations['safety_margin']]
        
        fig3 = go.Figure(go.Bar(
            x=top_recommendations['crop_name'],
            y=top_recommendations['safety_margin'],
            marker_color=colors,
            text=top_recommendations['safety_margin'].apply(lambda x: f"{x:+.1f} ℃"),
            textposition='outside',
            textfont=dict(size=12, color='#374151')
        ))
        
        fig3.add_hline(
            y=0, 
            line_dash="solid", 
            line_color="#6B7280", 
            line_width=1.5,
            annotation_text="동해 위험 기준선 (0℃)",
            annotation_position="bottom right"
        )
        
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=520,
            bargap=0.45,
            xaxis=dict(tickfont=dict(size=13, color="#374151"), showgrid=False),
            yaxis=dict(
                title=dict(text="여유 온도 (℃)", font=dict(size=13, color="#374151")),
                gridcolor='rgba(156, 163, 175, 0.2)'
            ),
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig3, use_container_width=True)