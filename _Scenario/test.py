# Streamlit 애플리케이션
import streamlit as st
import pandas as pd
from model import WildfirePredictor

# 데이터 로드
train_data = pd.read_csv('data/final_output_train.csv')
test_data = pd.read_csv('data/final_output_test.csv')

# WildfirePredictor 인스턴스 생성
target_columns = ['DURATION_MIN', 'MBLZ_FFPWR_CNT', 'LYCRG_FIREMAN_CNT']
predictor = WildfirePredictor(train_data, test_data, target_columns)
predictor.preprocess_data()
predictor.fit_models()

# Streamlit 애플리케이션 시작
st.markdown("<h1 style='text-align: center; '>산불 피해 예측 시뮬레이션</h1>", unsafe_allow_html=True)
st.image('img/mountain.jpg')

# 사이드바 입력
st.sidebar.header("산불 아이디 입력")

# Session state로 버튼 클릭 상태 관리
if "analyze_clicked" not in st.session_state:
    st.session_state.analyze_clicked = False  # 초기값 설정

# OBJT_ID 선택 드롭다운
objt_id_options = ['(입력)'] + list(test_data['OBJT_ID'].unique())
selected_objt_id = st.sidebar.selectbox("산불 아이디 선택", objt_id_options)

# "분석 실행" 버튼
if st.sidebar.button("분석 실행"):
    st.session_state.analyze_clicked = True  # 버튼 클릭 상태 업데이트

# 분석 작업
if st.session_state.analyze_clicked:
    is_custom_input = selected_objt_id == '(입력)'
    user_inputs = {}

    if is_custom_input:
        st.sidebar.markdown("직접 요소를 입력하세요.")
        st.sidebar.text_input("위치", "강원도", disabled=True)
        # 사용자가 직접 입력한 값 처리
    else:
        selected_row = test_data[test_data['OBJT_ID'] == int(selected_objt_id)]
        if selected_row.empty:
            st.warning("선택된 OBJT_ID가 없습니다. 가장 유사한 데이터로 대체합니다.")
            selected_row = predictor.get_similar_row(test_data[predictor.feature_columns])
        user_inputs = selected_row.iloc[0].to_dict()

    # 예측 수행
    input_row = pd.DataFrame([user_inputs])
    X_input = input_row[predictor.feature_columns].values
    predictions = predictor.predict(X_input)

    # 결과 출력
    st.markdown("<h2 style='text-align: center; '>예측 결과</h2>", unsafe_allow_html=True)
    st.write(f"⏳ **예상 진화 시간**: {predictions['DURATION_MIN']:.2f} 분")
    st.write(f"💧 **예상 소방 설비 사용량**: {predictions['MBLZ_FFPWR_CNT']:.2f} 대")
    st.write(f"👩‍🚒 **예상 소방인력 수**: {predictions['LYCRG_FIREMAN_CNT']:.2f} 명")
else:
    st.info("OBJT_ID를 선택하거나 입력값을 설정한 후, '분석 실행' 버튼을 눌러 예측을 시작하세요.")
