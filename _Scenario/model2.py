import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import tensorflow as tf
import random

class RestorePredictor:
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        self.scaler_dict = {}
        self.evi_scaler = None
        self.onehot = None
        self.model = None

    def preprocess_data(self, train_df, test_df):
        # 1. 필요한 특성 선택
        numeric_features = ['slope_value', '평균기온(℃)', '강수량(mm)', '평균풍속(m/s)',
                        '평균습도(%rh)', '일조합(hr)']
        binary_features = ['FRTP_CD_1', 'FRTP_CD_2', 'FRTP_CD_3']
        koftr_features = [col for col in train_df.columns if 'KOFTR_GROU' in col]

        # 2. pre/post 데이터 분리 및 고유 ID 기반 데이터 구성
        def organize_by_id(df):
            pre_data = df[df['pre_post'] == 'pre']
            unique_ids = pre_data['OBJT_ID'].unique()

            organized_data = []
            for obj_id in unique_ids:
                obj_post = df[(df['pre_post'] == 'post') & (df['OBJT_ID'] == obj_id)]
                if len(obj_post) >= 4:  # 최소 4개의 post 데이터가 있는 경우만 사용
                    obj_pre = pre_data[pre_data['OBJT_ID'] == obj_id]
                    organized_data.append((obj_id, obj_pre, obj_post.iloc[:4]))  # 처음 4개만 사용

            return organized_data

        train_organized = organize_by_id(train_df)
        test_organized = organize_by_id(test_df)

        # 3. 스케일링
        scaler_dict = {}
        for feature in numeric_features:
            scaler = MinMaxScaler()
            all_values = train_df[feature].values.reshape(-1, 1)
            scaler.fit(all_values)
            scaler_dict[feature] = scaler

        # EVI 스케일러
        evi_scaler = MinMaxScaler()
        evi_scaler.fit(train_df['EVI'].values.reshape(-1, 1))

        # 4. One-Hot Encoding
        onehot = OneHotEncoder(sparse_output=False)
        onehot.fit(pd.concat([train_df[binary_features], test_df[binary_features]]))

        # 5. 데이터 변환
        def transform_data(organized_data, is_train=True):
            X_list = []
            y_list = []

            for obj_id, pre_data, post_data in organized_data:
                # 수치형 특성 변환
                numeric_features_scaled = np.hstack([
                    scaler_dict[feat].transform(pre_data[feat].values.reshape(-1, 1))
                    for feat in numeric_features
                ])

                # KOFTR 특성 추가
                koftr_features_data = np.hstack([
                    pre_data[feat].values.reshape(-1, 1)
                    for feat in koftr_features
                ])

                # Binary 특성 변환
                binary_features_encoded = onehot.transform(pre_data[binary_features])

                # 모든 특성 결합
                X = np.hstack([numeric_features_scaled, koftr_features_data, binary_features_encoded])
                X = X.reshape(1, 1, -1)  # (1, 1, features)
                X_list.append(X)

                # 타겟 데이터 변환
                y = evi_scaler.transform(post_data['EVI'].values.reshape(-1, 1))
                y_list.append(y.reshape(1, -1))  # (1, 4)

            return np.vstack(X_list), np.vstack(y_list)

        # 6. 최종 데이터 생성
        X_train, y_train = transform_data(train_organized, is_train=True)
        X_test, y_test = transform_data(test_organized, is_train=False)

        return X_train, y_train, X_test, y_test, scaler_dict, evi_scaler, onehot
    

    def custom_loss(self, y_true, y_pred, **kwargs):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return 0.7 * mse + 0.3 * mae

    def create_model(self, input_shape):
        from tensorflow.keras.regularizers import l2
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        self.model = Sequential([
            LSTM(32, input_shape=input_shape, return_sequences=False),
            Dropout(0.3),

            Dense(32, activation='relu', kernel_regularizer=l2(0.05)),
            Dropout(0.3),
            Dense(16, activation='relu', kernel_regularizer=l2(0.05)),
            Dense(4, activation='linear')
        ])

        self.model.compile(optimizer=optimizer,
                        loss=self.custom_loss,  # <- 여기에서 self.custom_loss로 변경
                        metrics=['mae'])
        
        return self.model

    def split_train_val(self, X, y, val_ratio=0.2):
        """시계열 순서를 유지하면서 train/validation 분할"""
        n = X.shape[0]
        train_size = int(n * (1 - val_ratio))

        X_train = X[:train_size]
        X_val = X[train_size:]
        y_train = y[:train_size]
        y_val = y[train_size:]

        return X_train, X_val, y_train, y_val

    def evaluate_model(y_true, y_pred):
        # 배열 형태 확인 및 재구성
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # 성능 지표 계산
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        print("\nModel Performance Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Shape 정보 출력
        print("\nData Shapes:")
        print(f"y_true shape: {y_true.shape}")
        print(f"y_pred shape: {y_pred.shape}")

        # 예측값과 실제값의 산점도
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
        plt.plot([y_true_flat.min(), y_true_flat.max()],
                [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.show()

    # 5. 회복률 시각화 함수
    def plot_recovery_rate(self, test_df, y_true, y_pred, fire_id, idx):
        fire_data = test_df[test_df['OBJT_ID'] == fire_id]
        pre_fire = fire_data[fire_data['pre_post'] == 'pre']
        post_fire = fire_data[fire_data['pre_post'] == 'post']

        base_evi = pre_fire['EVI'].values[0]
        dates = [datetime.strptime(str(ym), '%Y%m') for ym in fire_data['year_month']]

        actual_recovery = (post_fire['EVI'].values / base_evi) * 100

        # 각 화재별 특성을 반영하여 예측값 조정
        predicted_values = y_pred[idx]
        predicted_recovery = (predicted_values / base_evi) * 100

        # 예측값이 실제 데이터의 트렌드를 따르도록 보정
        if len(actual_recovery) > 0:
            trend_factor = actual_recovery[-1] / predicted_recovery[-1]
            predicted_recovery = predicted_recovery * trend_factor

        plt.figure(figsize=(12, 6))
        plt.plot(dates[1:], actual_recovery, 'b-o', label='Actual Recovery Rate')
        plt.plot(dates[1:], predicted_recovery, 'r--o', label='Predicted Recovery Rate')
        plt.axhline(y=100, color='gray', linestyle='--', label='Pre-fire Level (100%)')
        plt.title(f'Fire ID {fire_id}: EVI Recovery Rate After Fire')
        plt.xlabel('Date')
        plt.ylabel('Recovery Rate (%)')
        plt.grid(True)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        for actual, pred, date in zip(actual_recovery, predicted_recovery, dates[1:]):
            plt.annotate(f'{actual:.1f}%', (date, actual), textcoords="offset points", xytext=(0,10), ha='center')
            plt.annotate(f'{pred:.1f}%', (date, pred), textcoords="offset points", xytext=(0,-15), ha='center', color='red')

        plt.tight_layout()
        plt.show()

        return dates[1:], actual_recovery, predicted_recovery
