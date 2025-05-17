#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
import xgboost as xgb 
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report 
import time 
 
def main(): 
    """
    풍속 변수가 추가된 데이터를 사용하여 기본 XGBoost 모델 훈련
    """ 
    start_time = time.time() 
    print("=== 산불 예측 모델 학습 시작 ===") 
 
    # 결과 디렉토리 생성 
    model_dir = "../../outputs/models" 
    os.makedirs(model_dir, exist_ok=True) 
    os.makedirs(os.path.join(model_dir, "plots"), exist_ok=True) 
 
    # 데이터 로드 
    input_file = "../../outputs/data/weather_data_with_wind.csv" 
    print(f"데이터 로드 중: {input_file}") 
    df = pd.read_csv(input_file, parse_dates=['acq_date']) 
    print(f"로드 완료: {len(df)}행, {len(df.columns)}열") 
 
    # 기본 탐색적 분석 
    print("\n=== 데이터 탐색 ===") 
    print(f"컬럼 목록: {list(df.columns)}") 
    print(f"기간: {df.acq_date.min()} ~ {df.acq_date.max()}") 
    print(f"산불 발생 비율: {df.af_flag.mean()*100:.4f}% ({df.af_flag.sum()}/{len(df)})") 
 
    # 피처 선택 (날씨 변수 + 풍속, 10u/10v는 제외) 
    features = ['t2m', 'td2m', 'tp', 'wind10m'] 
    target = 'af_flag' 
 
    # 결측치 확인 및 처리 
    missing = df[features].isnull().sum() 
    if missing.sum() > 0: 
        print(f"\n결측치 발견: \n{missing[missing > 0]}") 
        df = df.dropna(subset=features) 
        print(f"결측치 제거 후 데이터 크기: {len(df)}") 
    
    # 학습용/테스트용 분할 
    X = df[features] 
    y = df[target] 
    
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=42, stratify=y 
    ) 
    
    print(f"\n학습 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}") 
    print(f"학습 데이터 산불 비율: {y_train.mean()*100:.4f}%") 
    print(f"테스트 데이터 산불 비율: {y_test.mean()*100:.4f}%") 
 
    # XGBoost 모델 학습 
    print("\n=== 모델 학습 ===") 
 
    # 클래스 불균형 처리 
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() 
    print(f"불균형 보정 가중치(scale_pos_weight): {scale_pos_weight:.2f}") 
 
    # 모델 정의 
    model = xgb.XGBClassifier( 
        objective='binary:logistic', 
        scale_pos_weight=scale_pos_weight, 
        learning_rate=0.05, 
        n_estimators=100, 
        max_depth=5, 
        min_child_weight=1, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42 
    ) 
    
    # 학습 
    model.fit( 
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)], 
        eval_metric='auc', 
        early_stopping_rounds=10, 
        verbose=True 
    ) 
    
    # 모델 평가 
    print("\n=== 모델 평가 ===") 
    y_pred_proba = model.predict_proba(X_test)[:, 1] 
    y_pred = model.predict(X_test) 
    
    # ROC AUC 점수 
    auc_score = roc_auc_score(y_test, y_pred_proba) 
    print(f"ROC AUC 점수: {auc_score:.4f}") 
    
    # 혼동 행렬 
    cm = confusion_matrix(y_test, y_pred) 
    print("\n혼동 행렬:") 
    print(cm) 
    
    # 분류 보고서 
    print("\n분류 보고서:") 
    print(classification_report(y_test, y_pred)) 
    
    # 특성 중요도 
    print("\n특성 중요도:") 
    importance = model.feature_importances_ 
    for i, feat in enumerate(features): 
        print(f"{feat}: {importance[i]:.4f}") 
    
    # 특성 중요도 시각화 
    plt.figure(figsize=(10, 6)) 
    xgb.plot_importance(model, max_num_features=len(features)) 
    plt.title("특성 중요도") 
    plt.tight_layout() 
    plt.savefig(f"{model_dir}/plots/feature_importance.png") 
    print(f"특성 중요도 시각화 저장: {model_dir}/plots/feature_importance.png") 
    
    # 모델 저장 
    model_file = f"{model_dir}/xgboost_weather_model.json" 
    model.save_model(model_file) 
    print(f"모델 저장 완료: {model_file}") 
    
    # 소요 시간 출력 
    processing_time = time.time() - start_time 
    print(f"\n총 처리 시간: {processing_time:.2f}초") 
    print("=== 모델 학습 완료 ===") 
 
if __name__ == "__main__": 
    main() 
