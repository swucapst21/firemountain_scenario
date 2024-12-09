import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


class WildfirePredictor:
    def __init__(self, train_data, test_data, target_columns):
        """
        산불 피해 예측 모델링 클래스.
        :param train_data: 학습용 데이터프레임.
        :param test_data: 테스트 데이터프레임.
        :param target_columns: 타겟 변수 리스트.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.target_columns = target_columns
        self.feature_columns = [col for col in train_data.columns if col not in target_columns + ['OBJT_ID']]
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.models = {}  # 각 타겟 변수별 모델 저장 공간
    
    def preprocess_data(self):
        """결측값 처리 및 전처리 수행."""
        self.train_features = self.knn_imputer.fit_transform(self.train_data[self.feature_columns])
        self.test_features = self.knn_imputer.transform(self.test_data[self.feature_columns])
    
    def fit_models(self):
        """
        모델 학습.
        """
        # 모델 정의
        self.models = {
            'DURATION_MIN': StackingRegressor(estimators=[
                ('ridge', Ridge()),
                ('extra_trees', ExtraTreesRegressor(
                    n_estimators=88, max_depth=17, random_state=42)),
                ('svr', SVR()),
                ('decision_tree', DecisionTreeRegressor(
                    max_depth=17, random_state=42)),
                ('random_forest', RandomForestRegressor(
                    n_estimators=88, max_depth=17, min_samples_split=9, random_state=42))
            ], final_estimator=Ridge()),

            'MBLZ_FFPWR_CNT': VotingRegressor(estimators=[
                ('ridge', Ridge(alpha=0.003253527491118228)),
                ('random_forest', RandomForestRegressor(
                    n_estimators=300, max_depth=7, min_samples_split=9, min_samples_leaf=4, random_state=62))
            ]),

            'LYCRG_FIREMAN_CNT': VotingRegressor(estimators=[
                ('ridge', Ridge()),
                ('extra_trees', ExtraTreesRegressor(
                    n_estimators=446, max_depth=8, random_state=54)),
                ('gradient_boosting', GradientBoostingRegressor(
                    n_estimators=446, max_depth=8, learning_rate=0.1693, random_state=54)),
                ('random_forest', RandomForestRegressor(
                    n_estimators=446, max_depth=8, random_state=54))
            ])
        }

        # 모델 학습
        for target in self.target_columns:
            self.models[target].fit(self.train_features, self.train_data[target])

    def predict(self, input_data):
        """
        입력 데이터로 예측 수행.
        :param input_data: 예측용 입력 데이터 (numpy array).
        :return: 타겟 변수별 예측값 딕셔너리.
        """
        predictions = {}
        for target in self.target_columns:
            predictions[target] = self.models[target].predict(input_data)[0]
        return predictions

    def get_similar_row(self, input_row):
        """
        입력 데이터와 가장 유사한 테스트 데이터 행을 찾아 반환.
        :param input_row: 예측에 사용할 입력 데이터 (numpy array).
        :return: 가장 유사한 테스트 데이터 행 (DataFrame).
        """
        # 유클리드 거리 계산
        distances = euclidean_distances(input_row, self.test_features)
        
        # 가장 가까운 행의 인덱스 찾기
        most_similar_idx = np.argmin(distances)
        
        # 가장 유사한 행 반환
        return self.test_data.iloc[[most_similar_idx]]

    def predict_with_similar_row(self, input_data):
        """
        입력 데이터와 가장 유사한 테스트 데이터 행으로 예측 수행.
        :param input_data: 예측에 사용할 입력 데이터 (numpy array).
        :return: 타겟 변수별 예측값 딕셔너리.
        """
        # 가장 유사한 행 가져오기
        similar_row = self.get_similar_row(input_data)
        
        # 유사한 행의 feature 데이터 추출
        similar_features = self.test_features[self.test_data.index == similar_row.index[0]]
        
        # 예측 수행
        predictions = {}
        for target in self.target_columns:
            predictions[target] = self.models[target].predict(similar_features)[0]
        
        return predictions
