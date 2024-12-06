import pandas as pd
import streamlit as st
from model import WildfirePredictor

# 데이터 로드
train_data = pd.read_csv('data/final_output_train.csv')
test_data = pd.read_csv('data/final_output_test.csv')

# WildfirePredictor 인스턴스 생성
target_columns = ['DURATION_MIN', 'MBLZ_FFPWR_CNT', 'LYCRG_FIREMAN_CNT']
predictor = WildfirePredictor(train_data, test_data, target_columns)
predictor.preprocess_data()
predictor.fit_models()

# Streamlit 애플리케이션
st.title("산불 피해 예측 시뮬레이션")
st.sidebar.header("산불 아이디 입력")

# OBJT_ID 선택 드롭다운
objt_id_options = ['(입력)'] + list(test_data['OBJT_ID'].unique())
selected_objt_id = st.sidebar.selectbox("산불 아이디 선택", objt_id_options)

# 입력 가능 여부 설정
is_custom_input = selected_objt_id == '(입력)'

# 원핫인코딩 컬럼 추출
onehot_prefixes = ['SIGUNGU_NM_', 'SPCNWS_CN_', 'WETHR_', 'OCCU_DAY_', 'IGN_BHF_']
onehot_columns_by_prefix = {
    prefix: [col for col in test_data.columns if col.startswith(prefix)]
    for prefix in onehot_prefixes
}

# 사용자 입력값 처리
user_inputs = {}

# 컬럼명을 사용자 친화적인 이름으로 매핑
friendly_names = {
    'SIGUNGU_NM_': '시군구명',
    'SPCNWS_CN_': '기상 특보 유형',
    'WETHR_': '날씨',
    'OCCU_DAY_': '요일',
    'IGN_BHF_': '화재 발생 위치'
}

# 사용자 입력값 처리
if is_custom_input:
    st.sidebar.markdown("직접 요소를 입력하세요.")
    st.sidebar.text_input("위치", "강원도", disabled=True)
    for prefix, columns in onehot_columns_by_prefix.items():
        if columns:
            # 사용자 친화적인 이름을 사용
            friendly_title = friendly_names.get(prefix, prefix.replace('_', ' ').title())
            dropdown_options = {col: col.replace(prefix, '') for col in columns}
            selected_value = st.sidebar.selectbox(
                f"{friendly_title} 선택", list(dropdown_options.values())
            )
            # 선택된 값을 원핫인코딩 형식으로 변환
            for col, value in dropdown_options.items():
                user_inputs[col] = 1 if value == selected_value else 0
else:
    st.sidebar.markdown("입력된 산불 아이디의 예측을 수행합니다.")
    selected_row = test_data[test_data['OBJT_ID'] == int(selected_objt_id)]
    
    if selected_row.empty:
        st.warning("선택된 OBJT_ID가 없습니다. 가장 유사한 데이터로 대체합니다.")
        selected_row = predictor.get_similar_row(test_data[predictor.feature_columns])
    user_inputs = selected_row.iloc[0][
        [col for cols in onehot_columns_by_prefix.values() for col in cols]
    ].to_dict()

# 모든 원핫인코딩 컬럼 기본값 추가 (결측 방지)
for prefix, columns in onehot_columns_by_prefix.items():
    for col in columns:
        if col not in user_inputs:
            user_inputs[col] = 0  # 기본값은 0으로 설정

# 분석 버튼
if st.sidebar.button("분석 실행"):
    if is_custom_input:
        input_row = pd.DataFrame([user_inputs])
    else:
        input_row = selected_row

    # 입력 데이터 준비
    missing_columns = [col for col in predictor.feature_columns if col not in input_row.columns]
    for col in missing_columns:
        input_row[col] = 0  # 결측된 열은 기본값 0으로 설정

    X_input = input_row[predictor.feature_columns].values

    # 예측 수행
    predictions = predictor.predict(X_input)

    # # 결과 출력
    # st.header("선택된 데이터")
    # st.write(input_row)

    st.header("예측 결과")
    st.write(f"📍 **위치**: 강원도")
    st.write(f"⏳ **예상 진화 시간**: {predictions['DURATION_MIN']:.2f} 분")
    st.write(f"💧 **예상 소방 설비 사용량**: {predictions['MBLZ_FFPWR_CNT']:.2f} 대")
    st.write(f"👩‍🚒 **예상 소방인력 수**: {predictions['LYCRG_FIREMAN_CNT']:.2f} 명")
else:
    st.info("OBJT_ID를 선택하거나 입력값을 설정한 후, '분석 실행' 버튼을 눌러 예측을 시작하세요.")
