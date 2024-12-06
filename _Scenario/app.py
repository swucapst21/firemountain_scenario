import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'  # Windows용 Malgun Gothic 폰트 경로
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# Sidebar for model selection
st.sidebar.title("모델 선택")
model_option = st.sidebar.selectbox("모델을 선택하세요", ["Model-1", "Model-2"])

# Model-1: CD2_Modeling_MSY
if model_option == "Model-1":
    st.title("Model-1: 산불 피해 예측")
    try:
        df = pd.read_csv("gangwon_drop_duplicate.csv")
    except FileNotFoundError:
        st.error("데이터 파일 'gangwon_drop_duplicate.csv'를 찾을 수 없습니다. 파일 경로를 확인하세요.")
        st.stop()

    # 데이터 준비
    target_columns = ['DURATION_MIN', 'MBLZ_FFPWR_CNT', 'LYCRG_FIREMAN_CNT']
    X = df.select_dtypes(exclude=['object']).drop(columns=target_columns)
    y = df[target_columns]

    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # 사용자 입력
    damage_area = st.sidebar.number_input("초기 피해 규모 (헥타르)", min_value=1, value=100)
    fireman_count = st.sidebar.number_input("소방 인력 수", min_value=1, value=50)

    if st.sidebar.button("예측 실행"):
        # 입력값 준비
        X_input = np.full((1, X_train.shape[1]), np.nan)
        X_input[:, 0] = damage_area
        X_input[:, 1] = fireman_count
        X_input = pd.DataFrame(X_input, columns=X.columns)  # Feature names 추가
        X_input = knn_imputer.transform(pd.concat([X, X_input]))[-1].reshape(1, -1)

        # 예측 수행
        predictions = rf.predict(X_input)

        # 결과 출력
        st.markdown(
            f"""
            <div style="background-color:#f9f9f9;padding:10px;border-radius:5px;margin:10px 0;">
                <h3>예측 결과</h3>
                <p><b>초기 피해 규모:</b> {damage_area} 헥타르</p>
                <p><b>소방 인력 수:</b> {fireman_count} 명</p>
                <p><b>예측 값:</b> {predictions.tolist()}</p>
            </div>
            """, unsafe_allow_html=True
        )

        # 예측 값 시각화
        st.subheader("예측 값 비교")
        fig, ax = plt.subplots()
        ax.bar(["DURATION_MIN", "MBLZ_FFPWR_CNT", "LYCRG_FIREMAN_CNT"], predictions[0])
        ax.set_ylabel("예측 값")
        ax.set_title("예측 결과 비교")
        st.pyplot(fig)

# Model-2: ConvLSTM
elif model_option == "Model-2":
    st.title("Model-2: ConvLSTM 기반 모델링")

    # 데이터 준비
    prefire_train_shape = (22, 10, 64, 64, 1)
    prefire_train = np.random.rand(*prefire_train_shape)
    postfire_train = np.random.rand(22, 1)

    # ConvLSTM 모델 정의
    model = Sequential([
        ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(10, 64, 64, 1)),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # 모델 학습
    history = model.fit(prefire_train, postfire_train, epochs=5, batch_size=2, verbose=0)

    st.write("ConvLSTM 모델이 학습되었습니다.")

    # 학습 손실 시각화
    st.subheader("학습 손실 추이")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ConvLSTM 학습 손실 추이")
    st.pyplot(fig)

    # 예측 예제
    test_data = np.random.rand(1, 10, 64, 64, 1)
    prediction = model.predict(test_data)
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:10px;border-radius:5px;margin:10px 0;">
            <h3>ConvLSTM 예측 결과</h3>
            <p><b>예측 값:</b> {prediction[0][0]:.2f}</p>
        </div>
        """, unsafe_allow_html=True
    )
