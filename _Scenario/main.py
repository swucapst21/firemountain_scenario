import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import os

# Hugging Face Token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face 토큰이 환경 변수에 설정되지 않았습니다.")
    st.stop()

# 데이터 로드
try:
    train_data = pd.read_csv("data/final_output_train.csv")
    test_data = pd.read_csv("data/final_output_test.csv")
except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    st.stop()

# 타겟 및 피처 분리
target_columns = ['DURATION_MIN', 'MBLZ_FFPWR_CNT', 'LYCRG_FIREMAN_CNT']
X_train = train_data.drop(columns=target_columns)
y_train = train_data[target_columns]
X_test = test_data.drop(columns=target_columns)
y_test = test_data[target_columns]

# Imputer 정의
knn_imputer = KNNImputer(n_neighbors=5)
X_train_imputed = knn_imputer.fit_transform(X_train)
X_test_imputed = knn_imputer.transform(X_test)

# 모델 정의
final_models = {
    'DURATION_MIN': StackingRegressor(estimators=[
        ('ridge', Ridge()),
        ('extra_trees', ExtraTreesRegressor(
            n_estimators=88, 
            max_depth=17, 
            random_state=42
        )),
        ('svr', SVR()),
        ('decision_tree', DecisionTreeRegressor(
            max_depth=17, 
            random_state=42
        )),
        ('random_forest', RandomForestRegressor(
            n_estimators=88, 
            max_depth=17, 
            min_samples_split=9, 
            random_state=42
        ))
    ], final_estimator=Ridge()),

    'MBLZ_FFPWR_CNT': VotingRegressor(estimators=[
        ('ridge', Ridge(alpha=0.003253527491118228)),
        ('random_forest', RandomForestRegressor(
            n_estimators=300, 
            max_depth=7, 
            min_samples_split=9, 
            min_samples_leaf=4, 
            random_state=62
        ))
    ]),

    'LYCRG_FIREMAN_CNT': VotingRegressor(estimators=[
        ('ridge', Ridge()),
        ('extra_trees', ExtraTreesRegressor(
            n_estimators=446, 
            max_depth=8, 
            random_state=54
        )),
        ('gradient_boosting', GradientBoostingRegressor(
            n_estimators=446, 
            max_depth=8, 
            learning_rate=0.1693, 
            random_state=54
        )),
        ('random_forest', RandomForestRegressor(
            n_estimators=446, 
            max_depth=8, 
            random_state=54
        ))
    ])
}

# 모델 학습
for target, model in final_models.items():
    model.fit(X_train_imputed, y_train[target])

# LLaMA3 모델 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", token=hf_token)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Streamlit App
st.title("산불 피해 예측 시뮬레이션")
st.sidebar.header("입력값 설정")

# 사용자 입력
disabled_region = st.sidebar.text_input("위치", "강원도", disabled=True)  # 입력 비활성화
damage_area = st.sidebar.number_input("초기 피해 규모 (헥타르)", min_value=1, value=100)
fireman_count = st.sidebar.number_input("소방 인력 수", min_value=1, value=50)

if st.sidebar.button("시뮬레이션 실행"):
    # 입력값 처리
    X_input = pd.DataFrame(
        np.full((1, X_train.shape[1]), np.nan),
        columns=X_train.columns
    )
    X_input.iloc[0, 0] = damage_area
    X_input.iloc[0, 1] = fireman_count

    # KNN Imputer로 나머지 값을 채움
    X_input_imputed = knn_imputer.transform(X_input)

    # 예측 수행
    y_pred_duration = final_models['DURATION_MIN'].predict(X_input_imputed)[0]
    y_pred_mblz = final_models['MBLZ_FFPWR_CNT'].predict(X_input_imputed)[0]
    y_pred_lycrg = final_models['LYCRG_FIREMAN_CNT'].predict(X_input_imputed)[0]

    # LLaMA3 요약 생성
    prompt = f"""
    산불 피해 예측 요약:
    - 위치: 강원도
    - 예상 피해 규모는 {damage_area}헥타르로 상당한 크기입니다.
    - 소방 인력 {fireman_count}명이 투입될 예정이며, 예상 진화 시간은 약 {y_pred_duration:.2f}분입니다.
    - 신속한 대응이 필요하며 추가 장비와 자원이 요구될 가능성이 있습니다.
    """
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    output = model.generate(input_ids, max_length=512, temperature=0.7)
    llama_summary = tokenizer.decode(output[0], skip_special_tokens=True)

    # 결과 출력
    st.header("예측 결과")
    st.write(f"📍 **위치**: 강원도")
    st.write(f"🔥 **초기 피해 규모**: {damage_area} 헥타르")
    st.write(f"🚒 **소방 인력**: {fireman_count}명")
    st.write(f"⏳ **예상 진화 시간**: {y_pred_duration:.2f} 분")
    st.write(f"💧 **예상 소방 설비 사용량**: {y_pred_mblz:.2f} 대")
    st.write(f"👩‍🚒 **예상 소방관 수**: {y_pred_lycrg:.2f} 명")

    st.header("시뮬레이션 요약")
    st.write(llama_summary)
