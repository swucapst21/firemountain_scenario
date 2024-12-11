import pandas as pd
import streamlit as st
from model1 import WildfirePredictor
from model2 import RestorePredictor  # RestorePredictor 클래스 가져오기
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.callbacks import EarlyStopping

def prepare_restore_predictor():
    """RestorePredictor를 준비하고 모델을 학습하는 함수"""
    train_df_cp = pd.read_csv('_Scenario/data/train_df_cp.csv', encoding='cp949')
    test_df_cp = pd.read_csv('_Scenario/data/test_df_cp.csv', encoding='cp949')

    # RestorePredictor 인스턴스 생성
    predictor = RestorePredictor(seed=42)

    # 데이터 전처리
    X_train, y_train, X_test, y_test, scalers, evi_scaler, onehot_encoder = predictor.preprocess_data(
        train_df_cp, test_df_cp
    )

    # 데이터 분할
    X_train_final, X_val, y_train_final, y_val = predictor.split_train_val(X_train, y_train)

    # 모델 생성 및 컴파일
    input_shape = (X_train_final.shape[1], X_train_final.shape[2])
    predictor.create_model(input_shape)

    # 학습 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    # 모델 학습
    predictor.model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1
    )

    # 테스트 데이터 예측
    y_pred = predictor.model.predict(X_test)

    return predictor, test_df_cp, y_test, y_pred

def plot_recovery_rate_streamlit(predictor, test_df_cp, y_test, y_pred, fire_id):
    """선택된 산불 ID에 대한 복원율 그래프를 Streamlit에 출력"""
    unique_fires = test_df_cp['OBJT_ID'].unique()
    idx = list(unique_fires).index(fire_id)

    buffer = BytesIO()
    predictor.plot_recovery_rate(test_df_cp, y_test, y_pred, fire_id, idx)
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    st.image(buffer, use_container_width=True)
    plt.close()

    st.markdown("<h3 style='text-align: center; '>[상세 복원 정보]</h3>", unsafe_allow_html=True)

    # RestorePredictor에서 날짜와 복원율 데이터 추출
    dates, actual_recovery, predicted_recovery = restore_predictor.plot_recovery_rate(
        test_df_cp, y_test, y_pred, fire_id, idx
    )

    # Streamlit에 날짜와 복원율 출력
    for date, actual, predicted in zip(dates, actual_recovery, predicted_recovery):
        st.write(f"🗓️ **날짜**: {date.strftime('%Y-%m')}")
        st.write(f"🕑 **실제 복원율**: {actual:.2f}%")
        st.write(f"🕣 **예측 복원율**: {predicted:.2f}%")
        st.markdown("---")

# 데이터 로드
train_data = pd.read_csv('_Scenario/data/final_train.csv')
test_data = pd.read_csv('_Scenario/data/final_test.csv')

# WildfirePredictor 인스턴스 생성
target_columns = ['DURATION_MIN', 'MBLZ_FFPWR_CNT', 'LYCRG_FIREMAN_CNT']
predictor = WildfirePredictor(train_data, test_data, target_columns)
predictor.preprocess_data()
predictor.fit_models()

# Streamlit 애플리케이션
st.markdown("<h1 style='text-align: center; '>산불 진화 자원 및 복원 예측 시나리오</h1>", unsafe_allow_html=True)
st.image('_Scenario/img/mountain.jpg')
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
        # Retrieve specific details for the selected OBJT_ID
        # Use user inputs for display
        location = "강원도"  # User-specified location
        location_sigungu = [col for col in user_inputs if col.startswith('SIGUNGU_NM_') and user_inputs[col] == 1]
        day = [col for col in user_inputs if col.startswith('OCCU_DAY_') and user_inputs[col] == 1]
        weather = [col for col in user_inputs if col.startswith('WETHR_') and user_inputs[col] == 1]
        special_notice = [col for col in user_inputs if col.startswith('SPCNWS_CN_') and user_inputs[col] == 1]
        ignition_place = [col for col in user_inputs if col.startswith('IGN_BHF_') and user_inputs[col] == 1]

        # Display details
        st.write(f"📍 **위치**: {location} {location_sigungu[0].replace('SIGUNGU_NM_', '') if location_sigungu else ''}")
        st.write(f"📅 **요일**: {day[0].replace('OCCU_DAY_', '') if day else ''}")
        st.write(f"☁️ **날씨**: {weather[0].replace('WETHR_', '') if weather else ''}")
        st.write(f"🌟 **기상특보 유형**: {special_notice[0].replace('SPCNWS_CN_', '') if special_notice else ''}")
        st.write(f"🔥 **화재 발생 위치**: {ignition_place[0].replace('IGN_BHF_', '') if ignition_place else ''}")

        # 데이터 부족 안내 메시지 출력
        st.markdown("<h3 style='text-align: center; '>[진화 자원 예측 결과]</h3>", unsafe_allow_html=True)
        st.write(f"⏳ **예상 진화 시간**: 데이터 부족으로 인한 예측 불가")
        st.write(f"💧 **예상 소방 설비 사용량**: 데이터 부족으로 인한 예측 불가")
        st.write(f"👩‍🚒 **예상 소방인력 수**: 데이터 부족으로 인한 예측 불가")
    else:
        input_row = selected_row
        # 입력 데이터 준비
        missing_columns = [col for col in predictor.feature_columns if col not in input_row.columns]
        for col in missing_columns:
            input_row[col] = 0  # 결측된 열은 기본값 0으로 설정
        X_input = input_row[predictor.feature_columns].values

        # 예측 수행
        predictions = predictor.predict(X_input)

        location = "강원도"  # Static example location
        location_sigungu = [col for col in selected_row.columns if col.startswith('SIGUNGU_NM_') and selected_row.iloc[0][col] == 1]
        day = [col for col in selected_row.columns if col.startswith('OCCU_DAY_') and selected_row.iloc[0][col] == 1]
        weather = [col for col in selected_row.columns if col.startswith('WETHR_') and selected_row.iloc[0][col] == 1]
        special_notice = [col for col in selected_row.columns if col.startswith('SPCNWS_CN_') and selected_row.iloc[0][col] == 1]
        ignition_place = [col for col in selected_row.columns if col.startswith('IGN_BHF_') and selected_row.iloc[0][col] == 1]

        st.write(f"📍 **위치**: {location} {location_sigungu[0].replace('SIGUNGU_NM_', '') if location_sigungu else ''}")
        st.write(f"📅 **요일**: {day[0].replace('OCCU_DAY_', '') if day else ''}")
        st.write(f"☁️ **날씨**: {weather[0].replace('WETHR_', '') if weather else ''}")
        st.write(f"🌟 **기상특보 유형**: {special_notice[0].replace('SPCNWS_CN_', '') if special_notice else ''}")
        st.write(f"🔥 **화재 발생 위치**: {ignition_place[0].replace('IGN_BHF_', '') if ignition_place else ''}")

        # 기존 예측 결과 출력
        st.markdown("<h3 style='text-align: center; '>[진화 자원 예측 결과]</h3>", unsafe_allow_html=True)
        st.write(f"⏳ **예상 진화 시간**: {predictions['DURATION_MIN']:.2f} 분")
        st.write(f"💧 **예상 소방 설비 사용량**: {predictions['MBLZ_FFPWR_CNT']:.2f} 대")
        st.write(f"👩‍🚒 **예상 소방인력 수**: {predictions['LYCRG_FIREMAN_CNT']:.2f} 명")

        # RestorePredictor 준비 및 예측 수행
        restore_predictor, test_df_cp, y_test, y_pred = prepare_restore_predictor()

        # 선택한 산불 ID의 복원율 그래프 출력
        fire_id = int(selected_objt_id)  # 입력된 산불 ID
        st.markdown(f"<h2 style='text-align: center;'>산불 ID {fire_id}: 복원 시나리오 시각화</h2>", unsafe_allow_html=True)
        plot_recovery_rate_streamlit(restore_predictor, test_df_cp, y_test, y_pred, fire_id)
            


else:
    st.info("OBJT_ID를 선택하거나 입력값을 설정한 후, '분석 실행' 버튼을 눌러 예측을 시작하세요.")
